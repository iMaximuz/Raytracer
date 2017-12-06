#pragma once

template<typename T>
struct is_scalar{
     static constexpr bool value = false;
};

#define Scalar_Trait(Type) template<> struct is_scalar<Type> { static constexpr bool value = true; };

#define Scalar_Trait_C(Type) Scalar_Trait(Type) \
                             Scalar_Trait(Type const) 

#define Scalar_Unsigned_Trait(base_type) Scalar_Trait_C(base_type) \
                                         Scalar_Trait_C(unsigned base_type) 

#define Scalar_Multisize_Trait(base_type) Scalar_Unsigned_Trait(base_type) \
                                          Scalar_Unsigned_Trait(short base_type) \
                                          Scalar_Unsigned_Trait(long base_type) 

Scalar_Unsigned_Trait(char)
Scalar_Multisize_Trait(int)
Scalar_Trait_C(float)
Scalar_Trait_C(double)
Scalar_Trait_C(long double)


