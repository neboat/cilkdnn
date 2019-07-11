#pragma once

#ifdef NOINLINEATTR
#define INLINEATTR __attribute__((noinline))
#else
#define INLINEATTR __attribute__((always_inline))
#endif
