// Exact double-coordinate memoization. Sorting preserves the Python batch order.
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <unordered_map>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
struct Key {
    std::uint64_t bits[3];
    bool operator==(const Key& other) const {
        return bits[0]==other.bits[0] && bits[1]==other.bits[1] && bits[2]==other.bits[2];
    }
};
static Key key(const double* point) { Key result; std::memcpy(result.bits,point,24); return result; }
struct Hash {
    std::size_t operator()(const Key& value) const {
        std::uint64_t hash=0x9e3779b97f4a7c15ULL;
        for(auto bits:value.bits) {
            bits^=bits>>30; bits*=0xbf58476d1ce4e5b9ULL;
            bits^=bits>>27; bits*=0x94d049bb133111ebULL; bits^=bits>>31;
            hash^=bits+0x9e3779b97f4a7c15ULL+(hash<<6)+(hash>>2);
        }
        return static_cast<std::size_t>(hash);
    }
};
using Cache=std::unordered_map<Key,double,Hash>;
API void* point_cache_create() { try { return new Cache; } catch(...) { return nullptr; } }
API void point_cache_destroy(void* handle) { delete static_cast<Cache*>(handle); }
API int point_cache_prepare(void* handle,std::int64_t count,const double* points,
        double* unique,double* values,std::int64_t* inverse,std::int64_t* missing,std::int64_t* counts) {
    try {
        auto& cache=*static_cast<Cache*>(handle);
        std::unordered_map<Key,std::int64_t,Hash> pending,zeros;
        std::int64_t misses=0;
        for(std::int64_t index=0;index<count;++index) {
            const double* point=points+3*index;
            const Key actual=key(point);
            // NumPy groups signed zeros numerically and uses the first row's
            // exact bits as the persistent key. Preserve that representative.
            bool has_zero=false;Key numeric=actual;
            for(int axis=0;axis<3;++axis) if(point[axis]==0) { numeric.bits[axis]=0;has_zero=true; }
            if(has_zero) {
                auto previous=zeros.emplace(numeric,index);
                if(!previous.second) { inverse[index]=previous.first->second;continue; }
            }
            auto found=cache.find(actual);
            inverse[index]=index;
            if(found!=cache.end()) { values[index]=found->second;continue; }
            if(!has_zero) {
                auto previous=pending.emplace(actual,index);
                if(!previous.second) { inverse[index]=previous.first->second;continue; }
            }
            std::memcpy(unique+3*index,point,24);
            values[index]=0;missing[misses++]=index;
        }
        // Only misses reach the evaluator. Keep precisely the original
        // lexicographic batch order, including its small-batch behavior.
        std::sort(missing,missing+misses,[&](auto a,auto b) {
            for(int axis=0;axis<3;++axis) {
                if(unique[3*a+axis]<unique[3*b+axis]) return true;
                if(unique[3*a+axis]>unique[3*b+axis]) return false;
            }
            return false;
        });
        counts[0]=count;counts[1]=misses;return 0;
    } catch(...) { return -1; }
}
API int point_cache_store(void* handle,std::int64_t count,const double* points,const double* values) {
    try {
        auto& cache=*static_cast<Cache*>(handle);
        for(std::int64_t i=0;i<count;++i) cache[key(points+3*i)]=values[i];
        return 0;
    } catch(...) { return -1; }
}
