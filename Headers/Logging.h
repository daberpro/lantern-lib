#pragma once
#include "../pch.h"
#include "Vector.h"
#include "LanternString.h"
//#include "../FeedForwardNetwork/FeedForwardNetwork.h"


inline std::ostream &operator<<(std::ostream & _os,const af::array &_tensor)
{
    _os << af::toString("Tensor",_tensor,16,true);
    return _os;
}


template <typename T>
inline std::ostream &operator<<(std::ostream &_os, const lantern::utility::Vector<T> & _obj)
{

    _os << "[";
    for (size_t i = 0; i < _obj.size(); ++i) {
        if (i > 0) _os << ", ";
        if constexpr (std::is_same_v<T, af::array>) {
            _os << af::toString("Tensor", _obj[i], 16, true);
        }
        else {
            _os << _obj[i];
        }
    }
    _os << "]";
    return _os;
}


template <>
struct std::formatter<af::array> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const af::array& _obj, std::format_context& _ctx) const {
        return std::format_to(_ctx.out(), "{}", af::toString("Tensor",_obj,16,true));
    }
};

template <>
struct std::formatter<af::dim4> {

    constexpr auto parse(std::format_parse_context& _ctx)  noexcept {
        return _ctx.begin();
    }

    auto format(const af::dim4& _d, std::format_context& _ctx) const {
        // Use std::ostringstream because af::toString returns std::string
        return std::format_to(_ctx.out(), "[{}, {}, {}, {}]", _d[0], _d[1], _d[2], _d[3]);
    }
};

template <>
struct std::formatter<lantern::utility::Vector<af::array>> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const lantern::utility::Vector<af::array>& _obj, std::format_context& _ctx) const {
        
        std::ostringstream oss_;
        oss_ << "[";
        for (size_t i = 0; i < _obj.size(); ++i) {
            if (i > 0) oss_ << ", ";
            oss_ << af::toString("Tensor", _obj[i], 16, true);
        }
        oss_ << "]";
        return std::format_to(_ctx.out(), "{}", oss_.str());
    }
};

template <>
struct std::formatter<lantern::string::String> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const lantern::string::String& _obj, std::format_context& _ctx) const {
        return std::format_to(_ctx.out(), "{}", (std::string_view) _obj);
    }
};

template <>
struct std::formatter<lantern::utility::Vector<uint32_t>> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const lantern::utility::Vector<uint32_t>& _obj, std::format_context& _ctx) const {
        
        std::ostringstream oss_;
        oss_ << "[";
        for (size_t i = 0; i < _obj.size(); ++i) {
            if (i > 0) oss_ << ", ";
            oss_ << std::to_string(_obj[i]);
        }
        oss_ << "]";
        return std::format_to(_ctx.out(), "{}", oss_.str());
    }
};

template <>
struct std::formatter<lantern::utility::Vector<double>> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const lantern::utility::Vector<double>& _obj, std::format_context& _ctx) const {
        
        std::ostringstream oss_;
        oss_ << "[";
        for (size_t i = 0; i < _obj.size(); ++i) {
            if (i > 0) oss_ << ", ";
            oss_ << std::to_string(_obj[i]);
        }
        oss_ << "]";
        return std::format_to(_ctx.out(), "{}", oss_.str());
    }
};

template <>
struct std::formatter<lantern::utility::Vector<lantern::string::String>> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const lantern::utility::Vector<lantern::string::String>& _obj, std::format_context& _ctx) const {
        
        std::string result_ = "[";
        for(uint32_t i = 0; i < _obj.size(); i++){
            result_ += _obj[i];
            if(i != _obj.size() - 1) result_ += ",";
        }
        result_ += "]";
        return std::format_to(_ctx.out(), "{}", result_);

    }
};

template <>
struct std::formatter<lantern::utility::Vector<std::string>> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const lantern::utility::Vector<std::string>& _obj, std::format_context& _ctx) const {
        
        std::string result_ = "[";
        for(uint32_t i = 0; i < _obj.size(); i++){
            result_ += _obj[i];
            if(i != _obj.size() - 1) result_ += ",";
        }
        result_ += "]";
        return std::format_to(_ctx.out(), "{}", result_);

    }
};



template <>
struct std::formatter<lantern::utility::Vector<lantern::utility::Vector<lantern::string::String>>> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const lantern::utility::Vector<lantern::utility::Vector<lantern::string::String>>& _obj, std::format_context& _ctx) const {

        std::print("SIZE {}",_obj.size());
        std::ostringstream oss;
        oss << "[\n";
        for (uint32_t i = 0; i < _obj.size(); ++i) {
            oss << " [";
            for (uint32_t j = 0; j < _obj[i].size(); ++j) {
                oss << static_cast<std::string>(_obj[i][j]);
                if (j + 1 < _obj[i].size()) oss << ",";
            }
            oss << "] " << i;
            if (i + 1 < _obj.size()) oss << ",\n";
        }
        oss << "]";
        return std::format_to(_ctx.out(), "{}", oss.str());
    }
};


template <typename T>
struct std::formatter<lantern::utility::Vector<lantern::utility::Vector<T>>> {

    constexpr auto parse(std::format_parse_context& _ctx) {
        return _ctx.begin();
    }

    auto format(const lantern::utility::Vector<lantern::utility::Vector<T>>& _obj, std::format_context& _ctx) const {

        std::string out_ = "";
        if constexpr (std::is_same_v<T,std::string>) {

            out_ = "[\n";
            for (uint32_t i = 0; i < _obj.size(); i++) {
                for (uint32_t j = 0; j < _obj[i].size(); j++) {
                    if (j > 0) {
                        out_ += ", ";
                    }
                    out_ += _obj[i][j];
                }
                out_ += "\n";
            }
            out_ += "]\n";
        }
        else {
            std::ostringstream oss_;
            for (uint32_t i = 0; i < _obj.size(); i++) {
                for (uint32_t j = 0; j < _obj[i].size(); j++) {
                    if (j > 0) {
                        oss_ << ", ";
                    }
                    oss_ << std::to_string(_obj[i][j]);
                }
                oss_ << '\n';
            }
            out_ = oss_.str();
        }
        return std::format_to(_ctx.out(), "{}", out_);
    }
};
