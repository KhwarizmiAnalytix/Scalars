#pragma once

#include "common/scalars_export.h"

namespace scalars
{
/**
    //tex:
    //$$N(x)=\int_{-\infty}^{x}\frac{\exp{(-\frac{u^2}{2})}}{\sqrt{2\pi}}du$$
    **/
SCALARS_API double normalcdf(double z);

/**
    //tex:
    //$$N^{-1}(x)$$
    **/
SCALARS_API double inv_normalcdf(double p);
}  // namespace scalars
