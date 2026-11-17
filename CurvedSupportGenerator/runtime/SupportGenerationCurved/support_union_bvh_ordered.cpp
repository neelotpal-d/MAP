// Exact hard rounded-sweep union. Portable C ABI; model-offset subtraction
// stays in the Python adapter and uses the original distance implementation.
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
struct Primitive {
    double a[3], b[3], d[3], mid[3], lo[3], hi[3];
    double length2, length, ra, rb, sphere2,bucket2;
    int kind;
};
struct Node { double lo[3],hi[3],midlo[3],midhi[3]; int first,count,left,right; };
struct Tree {
    std::vector<Primitive> primitive;
    std::vector<int> order;
    std::vector<Node> nodes;
    double up[3];
    double pruning_slack=1e-12;
    int build(int first,int count) {
        Node n{}; n.first=first; n.count=count; n.left=n.right=-1;
        for(int k=0;k<3;++k) { n.lo[k]=n.midlo[k]=INFINITY; n.hi[k]=n.midhi[k]=-INFINITY; }
        for(int j=first;j<first+count;++j) for(int k=0;k<3;++k) {
            const auto &p=primitive[order[j]];
            n.lo[k]=std::min(n.lo[k],p.lo[k]); n.hi[k]=std::max(n.hi[k],p.hi[k]);
            n.midlo[k]=std::min(n.midlo[k],p.mid[k]); n.midhi[k]=std::max(n.midhi[k],p.mid[k]);
        }
        int id=int(nodes.size()); nodes.push_back(n);
        if(count>8) {
            int axis=0;
            for(int k=1;k<3;++k) if(n.hi[k]-n.lo[k]>n.hi[axis]-n.lo[axis]) axis=k;
            int half=count/2;
            std::nth_element(order.begin()+first,order.begin()+first+half,order.begin()+first+count,
                [&](int a,int b){return primitive[a].mid[axis]<primitive[b].mid[axis];});
            int left=build(first,half),right=build(first+half,count-half);
            nodes[id].left=left; nodes[id].right=right;
        }
        return id;
    }
    double box_lower(int id,const double *x) const {
        const auto& n=nodes[id];double lower=-INFINITY;
        for(int k=0;k<3;++k) lower=std::max(lower,std::max(n.lo[k]-x[k],x[k]-n.hi[k]));
        return lower;
    }
    double query(int id,const double *x,double global2,double best=INFINITY) const {
        const auto &n=nodes[id];
        if(global2>=0.) {
            double dist2=0.;
            for(int k=0;k<3;++k) {double d=std::max(0.,std::max(n.midlo[k]-x[k],x[k]-n.midhi[k]));dist2+=d*d;}
            if(dist2>global2) return best;
        } else for(int k=0;k<3;++k) if(x[k]<n.lo[k]-1e-9 || x[k]>n.hi[k]+1e-9) return best;
        if(box_lower(id,x)>best+pruning_slack) return best;
        if(n.left>=0) {
            int first=n.left,second=n.right;
            if(box_lower(first,x)>box_lower(second,x)) std::swap(first,second);
            best=query(first,x,global2,best);
            return query(second,x,global2,best);
        }
        for(int j=n.first;j<n.first+n.count;++j) {
            const auto &p=primitive[order[j]];
            bool skip=false; double sphere=0.,dot=0.;
            for(int k=0;k<3;++k) {
                if(x[k]<p.lo[k]-1e-9 || x[k]>p.hi[k]+1e-9) skip=true;
                double dm=x[k]-p.mid[k]; sphere+=dm*dm;
                dot+=(x[k]-p.a[k])*p.d[k];
            }
            if(global2>=0.) {if(sphere>global2) continue;}
            else if(skip || sphere>p.sphere2+1e-9 || sphere>p.bucket2) continue;
            double t=p.length2>1e-16?std::max(0.,std::min(1.,dot/p.length2)):0.;
            double radius=p.ra+t*(p.rb-p.ra);
            if(p.kind==2) {
                double s=std::max(0.,std::min(1.,std::min(t,1.-t)*p.length/3.));
                s=s*s*(3.-2.*s); radius=p.rb+(p.ra-p.rb)*s;
            }
            double norm=0.,lower=0.,upper=0.;
            for(int k=0;k<3;++k) {
                double dx=x[k]-p.a[k]-t*p.d[k]; norm+=dx*dx;
                lower+=(p.a[k]-x[k])*up[k]; upper+=(x[k]-p.b[k])*up[k];
            }
            double f=std::sqrt(norm)-radius;
            if(p.kind==3) f=std::max(f,std::max(lower,upper));
            best=std::min(best,f);
        }
        return best;
    }
    bool overlap(int id,const double *x,const double *half,double radius) const {
        const auto &n=nodes[id];
        for(int k=0;k<3;++k) if(x[k]+half[k]<n.lo[k]-1e-9 || x[k]-half[k]>n.hi[k]+1e-9) return false;
        if(n.left>=0) return overlap(n.left,x,half,radius)||overlap(n.right,x,half,radius);
        for(int j=n.first;j<n.first+n.count;++j) {
            const auto &p=primitive[order[j]];double norm=0.;bool keep=true;
            for(int k=0;k<3;++k) {
                double d=x[k]-p.mid[k];norm+=d*d;
                if(std::abs(d)>half[k]+(p.hi[k]-p.lo[k])*.5+1e-9) keep=false;
            }
            if(keep && std::sqrt(norm)<=radius+std::sqrt(p.sphere2)+1e-9) return true;
        }
        return false;
    }
};
API void* support_bvh_create(long long count,const double *rows,const double *up,const double *buckets) {
    try {
        auto *tree=new Tree; std::copy(up,up+3,tree->up);
        tree->primitive.resize(size_t(count)); tree->order.resize(size_t(count));
        std::iota(tree->order.begin(),tree->order.end(),0);
        for(int i=0;i<count;++i) {
            auto &p=tree->primitive[i]; const double *r=rows+9*i;
            p.ra=r[6];p.rb=r[7];p.kind=int(r[8]);p.length2=0.;
            for(int k=0;k<3;++k) {
                p.a[k]=r[k];p.b[k]=r[k+3];p.d[k]=p.b[k]-p.a[k];
                p.mid[k]=(p.a[k]+p.b[k])*.5;p.length2+=p.d[k]*p.d[k];
                double extent=std::abs(p.d[k])*.5+std::max(p.ra,p.rb);
                p.lo[k]=p.mid[k]-extent;p.hi[k]=p.mid[k]+extent;
                tree->pruning_slack=std::max(tree->pruning_slack,32*std::numeric_limits<double>::epsilon()*
                    std::max(std::abs(p.lo[k]),std::abs(p.hi[k])));
            }
            p.length=std::sqrt(p.length2); double radius=p.length*.5+std::max(p.ra,p.rb);
            p.sphere2=radius*radius;p.bucket2=(buckets[i]+1e-9)*(buckets[i]+1e-9);
        }
        tree->nodes.reserve(size_t(count)/2);tree->build(0,int(count));return tree;
    } catch(...) { return nullptr; }
}
API void support_bvh_destroy(void *handle) { delete static_cast<Tree*>(handle); }
API void support_bvh_evaluate(void *handle,long long count,const double *points,double *values,int threads,int batch,double global_radius) {
    const auto *tree=static_cast<const Tree*>(handle);
    #pragma omp parallel for num_threads(threads) if(count>2048)
    for(long long i=0;i<count;++i) {
        long long start=(i/batch)*batch;
        bool small=std::min((long long)batch,count-start)<64;
        values[i]=tree->query(0,points+3*i,small?(global_radius+1e-9)*(global_radius+1e-9):-1.);
    }
}
API void support_bvh_overlap(void *handle,long long count,const double *centers,const double *half,const double *radius,unsigned char *keep,int threads) {
    const auto *tree=static_cast<const Tree*>(handle);
    #pragma omp parallel for num_threads(threads) if(count>2048)
    for(long long i=0;i<count;++i) keep[i]=tree->overlap(0,centers+3*i,half+3*i,radius[i]);
}
