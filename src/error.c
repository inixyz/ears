#include "error.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static const char COLOR_GREEN[] = "\033[1;32m";
static const char COLOR_RED[] = "\033[1;31m";
static const char COLOR_RESET[] = "\033[0m";

static char *get_time_str(char *const time_str, const size_t time_str_len) {
  const time_t raw_time = time(NULL);
  const struct tm *const time_info = localtime(&raw_time);
  strftime(time_str, time_str_len, "%Y-%m-%d %H:%M:%S", time_info);
  return time_str;
}

static void log_message(FILE *const stream, const char *const color,
                        const char *const type, const char *const format,
                        va_list args) {
  const size_t TIME_STR_LEN = 20;
  char time_str[TIME_STR_LEN];
  fprintf(stream, "%s[%s][%s] ", color, type,
          get_time_str(time_str, TIME_STR_LEN));
  vfprintf(stream, format, args);
  fprintf(stream, "%s\n", COLOR_RESET);
}

void success(const char *const format, ...) {
  va_list args;
  va_start(args, format);
  log_message(stdout, COLOR_GREEN, "SUCCESS", format, args);
  va_end(args);
}

void error(const char *const format, ...) {
  va_list args;
  va_start(args, format);
  log_message(stderr, COLOR_RED, "ERROR", format, args);
  va_end(args);
  exit(EXIT_FAILURE);
}
