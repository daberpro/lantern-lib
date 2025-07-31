#pragma once
#include "../pch.h"
#include "Vector.h"
//#include "../FeedForwardNetwork/FeedForwardNetwork.h"


std::ostream &operator<<(std::ostream &os,const af::array &tensor)
{
    os << af::toString("Tensor",tensor,16,true);
    return os;
}


template <typename T>
std::ostream &operator<<(std::ostream &os, const lantern::utility::Vector<T> & obj)
{

    os << "[";
    for (size_t i = 0; i < obj.size(); ++i) {
        if (i > 0) os << ", ";
        if constexpr (std::is_same_v<T, af::array>) {
            os << af::toString("Tensor", obj[i], 16, true);
        }
        else {
            os << obj[i];
        }
    }
    os << "]";
    return os;
}

template <>
struct std::formatter<af::array> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const af::array& obj, std::format_context& ctx) const {
        return std::format_to(ctx.out(), "{}", af::toString("Tensor",obj,16,true));
    }
};

template <>
struct std::formatter<af::dim4> {

    constexpr auto parse(std::format_parse_context& ctx)  noexcept {
        return ctx.begin();
    }

    auto format(const af::dim4& d, std::format_context& ctx) const {
        // Use std::ostringstream because af::toString returns std::string
        return std::format_to(ctx.out(), "[{}, {}, {}, {}]", d[0], d[1], d[2], d[3]);
    }
};

template <>
struct std::formatter<lantern::utility::Vector<af::array>> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const lantern::utility::Vector<af::array>& obj, std::format_context& ctx) const {
        
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < obj.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << af::toString("Tensor", obj[i], 16, true);
        }
        oss << "]";
        return std::format_to(ctx.out(), "{}", oss.str());
    }
};
