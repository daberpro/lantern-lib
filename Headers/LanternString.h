#pragma once
#include "../pch.h"
#include "Utility.h"

namespace lantern
{

    namespace string
    {

        class String
        {
        private:
            std::string m_str;

        public:
            String(const char *_str) : m_str(_str) {}
            String(const std::string &_str) : m_str(_str) {}
            String(std::string &&_str) : m_str(std::move(_str)) {}

            std::string& Get()
            {
                return this->m_str;
            }

            const std::string& Get() const
            {
                return this->m_str;
            }

           
            explicit operator std::string(){
                return m_str;
            }

            explicit operator const std::string() const noexcept{
                return m_str;
            }

            operator std::string_view() const noexcept{
                return m_str;
            }

            template <typename T>
            T as() const
            {
                return lantern::utility::ConvertFromString<T>(this->m_str);
            }
        };

    }

}