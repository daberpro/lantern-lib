#pragma once
#include "../pch.h"
#include "Vector.h"
//#include "../FeedForwardNetwork/FeedForwardNetwork.h"


inline std::ostream &operator<<(std::ostream &os,const af::array &tensor)
{
    os << af::toString("Tensor",tensor,16,true);
    return os;
}


template <typename T>
inline std::ostream &operator<<(std::ostream &os, const lantern::utility::Vector<T> & obj)
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

template <>
struct std::formatter<lantern::utility::Vector<uint32_t>> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const lantern::utility::Vector<uint32_t>& obj, std::format_context& ctx) const {
        
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < obj.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << std::to_string(obj[i]);
        }
        oss << "]";
        return std::format_to(ctx.out(), "{}", oss.str());
    }
};

template <>
struct std::formatter<lantern::utility::Vector<double>> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const lantern::utility::Vector<double>& obj, std::format_context& ctx) const {
        
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < obj.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << std::to_string(obj[i]);
        }
        oss << "]";
        return std::format_to(ctx.out(), "{}", oss.str());
    }
};


template <typename T>
struct std::formatter<lantern::utility::Vector<lantern::utility::Vector<T>>> {

    constexpr auto parse(std::format_parse_context& ctx) {
        return ctx.begin();
    }

    auto format(const lantern::utility::Vector<lantern::utility::Vector<T>>& obj, std::format_context& ctx) const {

        std::string out = "";
        if constexpr (std::is_same_v<T,std::string>) {

            out = "[\n";
            for (uint32_t i = 0; i < obj.size(); i++) {
                for (uint32_t j = 0; j < obj[i].size(); j++) {
                    if (j > 0) {
                        out += ", ";
                    }
                    out += obj[i][j];
                }
                out += "\n";
            }
            out += "]\n";
        }
        else {
            std::ostringstream oss;
            for (uint32_t i = 0; i < obj.size(); i++) {
                for (uint32_t j = 0; j < obj[i].size(); j++) {
                    if (j > 0) {
                        oss << ", ";
                    }
                    oss << std::to_string(obj[i][j]);
                }
                oss << '\n';
            }
            out = oss.str();
        }
        return std::format_to(ctx.out(), "{}", out);
    }
};
