/*
 * KhwarizmiAnalytix Scalars library — DLL export/import (same pattern as LinearAlgebra).
 */
#pragma once

#define SCALARS_VISIBILITY_ENUM

#if defined(SCALARS_STATIC_DEFINE)
#define SCALARS_API
#define SCALARS_VISIBILITY
#define SCALARS_IMPORT
#define SCALARS_HIDDEN

#elif defined(SCALARS_SHARED_DEFINE)
#if defined(_WIN32) || defined(__CYGWIN__)
#ifdef SCALARS_BUILDING_DLL
#define SCALARS_API __declspec(dllexport)
#else
#define SCALARS_API __declspec(dllimport)
#endif
#define SCALARS_VISIBILITY
#define SCALARS_IMPORT __declspec(dllimport)
#define SCALARS_HIDDEN
#elif defined(__GNUC__) && __GNUC__ >= 4
#define SCALARS_API __attribute__((visibility("default")))
#define SCALARS_VISIBILITY __attribute__((visibility("default")))
#define SCALARS_IMPORT __attribute__((visibility("default")))
#define SCALARS_HIDDEN __attribute__((visibility("hidden")))
#else
#define SCALARS_API
#define SCALARS_VISIBILITY
#define SCALARS_IMPORT
#define SCALARS_HIDDEN
#endif

#else
#define SCALARS_API
#define SCALARS_VISIBILITY
#define SCALARS_IMPORT
#define SCALARS_HIDDEN
#endif
