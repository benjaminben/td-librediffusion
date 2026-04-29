// Debug logging for td-librediffusion. Routes to OutputDebugStringA on Windows
// so output appears in DebugView (Sysinternals) without needing a console.

#pragma once

#include <sstream>
#include <string>

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
  inline void td_debug_log_line(const std::string& s)
  {
    OutputDebugStringA(("[td-librediff] " + s + "\n").c_str());
  }
#else
  #include <cstdio>
  inline void td_debug_log_line(const std::string& s)
  {
    fprintf(stderr, "[td-librediff] %s\n", s.c_str());
    fflush(stderr);
  }
#endif

#define TDDBG(expr)                                                            \
  do                                                                           \
  {                                                                            \
    std::ostringstream _oss;                                                   \
    _oss << expr;                                                              \
    td_debug_log_line(_oss.str());                                             \
  } while(0)
