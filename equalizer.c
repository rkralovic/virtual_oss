/*-
 * Copyright (c) 2019 Google LLC, written by Richard Kralovic <riso@google.com>
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include <err.h>
#include <errno.h>
#include <fcntl.h>
#include <fftw3.h>
#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sysexits.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/soundcard.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "virtual_oss.h"

int in_background = 0;

static void Message(const char *fmt,...) {
  va_list list;
  if (in_background) return;
  va_start(list, fmt);
  vfprintf(stderr, fmt, list);
  va_end(list);
}

/* Masking window value for -1 < x < 1. Window must be symmetric, thus, this
 * function is queried for x >= 0 only.
 * Currently a Hann window. */
static double GetWindow(double x) {
  return 0.5 + 0.5 * cos(M_PI * x);
}

struct Equalizer {
  double rate;
  int block_size;

  double* fftw_time;  /* block_size * 2 elements, time domain */
  double* fftw_freq;  /* block_size * 2 elements, half-complex, freq domain */
  fftw_plan forward;
  fftw_plan inverse;
};

static int LoadFrequencyAmplification(struct Equalizer* e, const char* config) {
  double prev_f = 0.0;
  double prev_amp = 1.0;
  double next_f = 0.0;
  double next_amp = 1.0;

  for (int i = 0; i <= e->block_size / 2; ++i) {
    double f = e->rate / e->block_size * i;
    while (f >= next_f) {
      prev_f = next_f;
      prev_amp = next_amp;
      if (*config == 0) {
        next_f = e->rate;
        next_amp = prev_amp;
      } else {
        int len;
        if (sscanf(config, "%lf %lf %n", &next_f, &next_amp, &len) == 2) {
          config += len;
          if (next_f <= prev_f) {
            Message("Parse error: Nonincreasing sequence of frequencies.\n");
            return 0;
          }
        } else {
          Message("Parse error.\n");
          return 0;
        }
      }
      if (prev_f == 0.0) {
	prev_amp = next_amp;
      }
    }
    e->fftw_freq[i] =
     ((f - prev_f) / (next_f - prev_f)) * (next_amp - prev_amp) + prev_amp;
  }
  return 1;
}

static void EqualizerInit(struct Equalizer* e, double rate, int block_size) {
  e->rate = rate;
  e->block_size = block_size;

  int buffer_size = sizeof(double) * e->block_size;
  e->fftw_time = (double*)malloc(buffer_size);
  e->fftw_freq = (double*)malloc(buffer_size);

  e->forward = fftw_plan_r2r_1d(block_size, e->fftw_time, e->fftw_freq,
      FFTW_R2HC, FFTW_MEASURE);
  e->inverse = fftw_plan_r2r_1d(block_size, e->fftw_freq, e->fftw_time,
      FFTW_HC2R, FFTW_MEASURE);
}

static int EqualizerLoad(struct Equalizer* e, const char* config) {
  int retval = 0;
  int N = e->block_size;
  int buffer_size = sizeof(double) * N;

  memset(e->fftw_freq, 0, buffer_size);
  Message("\n\nReloading amplification specifications:\n%s\n", config);
  if (!LoadFrequencyAmplification(e, config)) goto end;

  double* requested_freq = (double*)malloc(buffer_size);
  memcpy(requested_freq, e->fftw_freq, buffer_size);

  fftw_execute(e->inverse);
  /* Multiply by symmetric window and shift */
  for (int i = 0; i < N / 2; ++i) {
    double weight = GetWindow(i / (double)(N / 2)) / N;
    e->fftw_time[N / 2 + i] = e->fftw_time[i] * weight;
  }
  for (int i = N / 2 - 1; i > 0; --i) {
    e->fftw_time[i] = e->fftw_time[N - i];
  }
  e->fftw_time[0] = 0;

  fftw_execute(e->forward);
  for (int i = 0; i < N; ++i) {
    e->fftw_freq[i] /= (double)N;
  }

  /* Debug output */
  for (int i = 0; i <= N / 2; ++i) {
    double f = (e->rate / N) * i;
    double a = sqrt(pow(e->fftw_freq[i], 2.0) +
        ((i > 0 && i < N / 2) ? pow(e->fftw_freq[N - i], 2.0) : 0));
    a *= N;
    double r = requested_freq[i];
    Message("%3.1lf Hz: requested %2.2lf, got %2.7lf (log10 = %.2lf), %3.7lfdb\n",
        f, r, a, log(a)/log(10), (log(a / r) / log(10.0)) * 10.0);
  }
  for (int i = 0; i < N; ++i) {
    Message("%.3lf ms: %.3lf\n", 1000.0 * i / e->rate, e->fftw_time[i]);
  }
  /* End of debug */

  retval = 1;

  free(requested_freq);
end:
  return retval;
}

static void EqualizerDone(struct Equalizer* e) {
  fftw_destroy_plan(e->forward);
  fftw_destroy_plan(e->inverse);
  free(e->fftw_time);
  free(e->fftw_freq);
}

static struct option equalizer_opts[] = {
  {"device", required_argument, NULL, 'd'},
  {"rate", required_argument, NULL, 'r'},
  {"block", required_argument, NULL, 'b'},
  {"channels", required_argument, NULL, 'c'},
  {"background", no_argument, NULL, 'B'},
  {"config", required_argument, NULL, 's'},
};

static void Usage() {
  Message("Usage: equalizer \n"
      "\t -d, --device [device]\n"
      "\t -r, --rate [rate in Hz, default 48000]\n"
      "\t -b, --block [block size in samples, default 2048]\n"
      "\t -c, --channels [channels, default 2]\n"
      "\t -B, --background\n"
      "\t -s, --config [equalizer configuration socket]\n");
  exit(EX_USAGE);
}

int main(int argc, char** argv) {
  struct Equalizer e;
  double rate = 48000.0;
  int block = 2048;
  int channels = 2;
  const char* socket_name = "/tmp/equalizer.socket";
  const char* dsp = "/dev/vdsp.ctl";
  int go_to_background = 0;

  int opt;
  while ((opt = getopt_long(argc, argv, "d:r:b:c:Bs:h", equalizer_opts, NULL))
      != -1) {
    switch (opt) {
    case 'd':
      dsp = optarg;
      break;
    case 'r':
      if (sscanf(optarg, "%lf", &rate) != 1) {
        Message("Cannot parse rate\n");
        Usage();
      }
      break;
    case 'b':
      block = strtol(optarg, NULL, 10);
      if (block == 0 || (block % 2)) {
        Message("Wrong block size\n");
        Usage();
      }
      break;
    case 'c':
      channels = strtol(optarg, NULL, 10);
      if (channels == 0) {
        Message("Wrong number of channels\n");
        Usage();
      }
      break;
    case 'B':
      go_to_background = 1;
      break;
    case 's':
      socket_name = optarg;
      break;
    default:
      Usage();
    }
  }

  if (go_to_background) {
    in_background = 1;
    if (daemon(0, 0) != 0) {
      errx(EX_SOFTWARE, "Cannot go to background");
    }
  }

  EqualizerInit(&e, rate, block);
  EqualizerLoad(&e, "");

  unlink(socket_name);
  int s = socket(AF_UNIX, SOCK_DGRAM, 0);
  if (s < 0) {
    errx(EX_SOFTWARE, "Cannot create socket");
  }

  struct sockaddr_un name;
  memset(&name, 0, sizeof(struct sockaddr_un));
  name.sun_family = AF_UNIX;
  strncpy(name.sun_path, socket_name, sizeof(name.sun_path) - 1);
  if (bind(s, (const struct sockaddr*)&name, sizeof(struct sockaddr_un))) {
    errx(EX_SOFTWARE, "Cannot bind socket");
  }
  while (1) {
    char buffer[65536];
    int len = read(s, buffer, sizeof(buffer) - 1);
    buffer[len] = 0;
    if (!EqualizerLoad(&e, buffer)) continue;

    int fd = open(dsp, O_RDWR);
    if (!fd) {
      Message("Cannot open device\n");
      continue;
    }

    struct virtual_oss_fir_filter fir = {};
    int error;

    for (fir.channel = 0; fir.channel < channels; ++fir.channel) {
      fir.number = 0;
      fir.filter_size = e.block_size;
      fir.filter_data = e.fftw_time;
      error = ioctl(fd, VIRTUAL_OSS_SET_TX_DEV_FIR_FILTER, &fir);
      if (error) {
        Message("Cannot set filter for channel %d: error %d, errno %d\n", fir.channel,
            error, errno);
      }
    }
    close(fd);
  }
  close(s);
  unlink(socket_name);
  EqualizerDone(&e);
  return 0;
}
