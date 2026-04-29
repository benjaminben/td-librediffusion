#include "dll_loader.hpp"
#include "td_debug_log.hpp"

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #include <windows.h>
#endif

namespace librediff
{

namespace
{
HMODULE g_librediff_module = nullptr;

std::wstring widen(const char* s)
{
    if(!s)
        return {};
    int len = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
    if(len <= 0)
        return {};
    std::wstring out(len - 1, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s, -1, out.data(), len);
    return out;
}

std::string narrow(const wchar_t* s)
{
    if(!s)
        return {};
    int len = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
    if(len <= 0)
        return {};
    std::string out(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, s, -1, out.data(), len, nullptr, nullptr);
    return out;
}

std::wstring last_error_string()
{
    DWORD err = GetLastError();
    LPWSTR buf = nullptr;
    DWORD n = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM
            | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, err, 0, reinterpret_cast<LPWSTR>(&buf), 0, nullptr);
    std::wstring out;
    if(n && buf)
    {
        out.assign(buf, n);
        // strip trailing newline
        while(!out.empty() && (out.back() == L'\r' || out.back() == L'\n'))
            out.pop_back();
        LocalFree(buf);
    }
    out += L" (code ";
    out += std::to_wstring(err);
    out += L")";
    return out;
}
} // namespace

bool ensure_libraries_loaded(std::string* err_out)
{
    if(g_librediff_module)
        return true;

    // Locate this DLL's directory.
    HMODULE self = nullptr;
    if(!GetModuleHandleExW(
           GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
               | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
           reinterpret_cast<LPCWSTR>(&ensure_libraries_loaded), &self))
    {
        if(err_out)
            *err_out = "GetModuleHandleEx for self failed";
        TDDBG("!! ensure_libraries_loaded: GetModuleHandleEx failed: "
              << narrow(last_error_string().c_str()));
        return false;
    }

    wchar_t path[MAX_PATH];
    DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
    if(n == 0 || n >= MAX_PATH)
    {
        if(err_out)
            *err_out = "GetModuleFileName failed or path too long";
        TDDBG("!! ensure_libraries_loaded: GetModuleFileName failed");
        return false;
    }

    // Strip filename to get directory.
    wchar_t* slash = path;
    for(wchar_t* p = path; *p; ++p)
        if(*p == L'\\' || *p == L'/')
            slash = p;
    *slash = L'\0';

    std::wstring dll_path = path;
    dll_path += L"\\librediffusion.dll";

    TDDBG("ensure_libraries_loaded: LoadLibraryEx " << narrow(dll_path.c_str()));

    // Restrict transitive dependency resolution to:
    //   - the directory of librediffusion.dll (= our plugin folder)
    //   - System32
    // This deliberately excludes the host application's directory so
    // TouchDesigner's bundled nvinfer_10.dll cannot win over our staged copy.
    HMODULE h = LoadLibraryExW(
        dll_path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);

    if(!h)
    {
        std::wstring msg = last_error_string();
        TDDBG("!! LoadLibraryEx librediffusion.dll failed: " << narrow(msg.c_str()));
        if(err_out)
            *err_out = "LoadLibraryEx librediffusion.dll: " + narrow(msg.c_str());
        return false;
    }

    g_librediff_module = h;
    TDDBG("ensure_libraries_loaded: librediffusion.dll loaded OK");
    return true;
}

} // namespace librediff
