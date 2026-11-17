// Standalone fixed-axis layer audit: exact triangle clipping against six box
// planes, finite nozzle poses, chronological layer ranks, shared read-only BVHs.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C" __attribute__((visibility("default")))
#endif
struct Node { double lo[3],hi[3]; int first,count,left=-1,right=-1,minrank,mask; };
struct Mesh { int first,count,rank,part; std::vector<int> order; std::vector<Node> nodes; };
struct Audit {
 const double* v; const int64_t* f; const int* face_ranks=nullptr; std::vector<Mesh> meshes;
 std::vector<int> order; std::vector<Node> nodes;
 const double* vertex(int face,int corner) const {return v+3*f[3LL*face+corner];}
 void bounds(Node& n) {for(int k=0;k<3;++k){n.lo[k]=INFINITY;n.hi[k]=-INFINITY;} n.mask=0;n.minrank=INT32_MAX;}
 int build_mesh(Mesh& m,int first,int count) {
  Node n{};bounds(n);n.first=first;n.count=count;
  if(face_ranks)for(int j=first;j<first+count;++j)n.minrank=std::min(n.minrank,face_ranks[m.order[j]]);
  for(int j=first;j<first+count;++j)for(int c=0;c<3;++c)for(int k=0;k<3;++k){double x=vertex(m.order[j],c)[k];n.lo[k]=std::min(n.lo[k],x);n.hi[k]=std::max(n.hi[k],x);}
  int id=(int)m.nodes.size();m.nodes.push_back(n);
  if(count>12){int axis=0;for(int k=1;k<3;++k)if(n.hi[k]-n.lo[k]>n.hi[axis]-n.lo[axis])axis=k;
   int half=count/2;std::nth_element(m.order.begin()+first,m.order.begin()+first+half,m.order.begin()+first+count,[&](int a,int b){double x=0,y=0;for(int c=0;c<3;++c){x+=vertex(a,c)[axis];y+=vertex(b,c)[axis];}return x<y;});
   int l=build_mesh(m,first,half),r=build_mesh(m,first+half,count-half);m.nodes[id].left=l;m.nodes[id].right=r;}
  return id;
 }
 int build_top(int first,int count) {
  Node n{};bounds(n);n.first=first;n.count=count;
  for(int j=first;j<first+count;++j){const auto&m=meshes[order[j]];const auto&b=m.nodes[0];n.minrank=std::min(n.minrank,m.rank);n.mask|=m.part;for(int k=0;k<3;++k){n.lo[k]=std::min(n.lo[k],b.lo[k]);n.hi[k]=std::max(n.hi[k],b.hi[k]);}}
  int id=(int)nodes.size();nodes.push_back(n);
  if(count>4){int axis=0;for(int k=1;k<3;++k)if(n.hi[k]-n.lo[k]>n.hi[axis]-n.lo[axis])axis=k;
   int half=count/2;std::nth_element(order.begin()+first,order.begin()+first+half,order.begin()+first+count,[&](int a,int b){const auto&x=meshes[a].nodes[0];const auto&y=meshes[b].nodes[0];return x.lo[axis]+x.hi[axis]<y.lo[axis]+y.hi[axis];});
   int l=build_top(first,half),r=build_top(first+half,count-half);nodes[id].left=l;nodes[id].right=r;}
  return id;
 }
 static bool overlap(const Node&n,const double*lo,const double*hi){for(int k=0;k<3;++k)if(n.hi[k]<lo[k]||n.lo[k]>hi[k])return false;return true;}
 bool clip(int face,const double*lo,const double*hi,double*w) const {
  double p[16][3],q[16][3];int size=3;
  for(int c=0;c<3;++c)for(int k=0;k<3;++k)p[c][k]=vertex(face,c)[k];
  for(int axis=0;axis<3;++axis)for(int side=0;side<2;++side){int next=0;double bound=side?hi[axis]:lo[axis];
   for(int j=0;j<size;++j){const double*a=p[j];const double*b=p[(j+1)%size];double da=side?a[axis]-bound:bound-a[axis];double db=side?b[axis]-bound:bound-b[axis];
    if(da<=0){for(int k=0;k<3;++k)q[next][k]=a[k];++next;}
    if((da<0&&db>0)||(da>0&&db<0)){double t=da/(da-db);for(int k=0;k<3;++k)q[next][k]=a[k]+t*(b[k]-a[k]);q[next][axis]=bound;++next;}}
   if(!next)return false;size=next;for(int j=0;j<size;++j)for(int k=0;k<3;++k)p[j][k]=q[j][k];}
  for(int k=0;k<3;++k){w[k]=0;for(int j=0;j<size;++j)w[k]+=p[j][k];w[k]/=size;}return true;
 }
 int query_mesh(const Mesh&m,int id,int rank,const double*lo,const double*hi,double*w) const {
  const auto&n=m.nodes[id];if((face_ranks&&n.minrank>=rank)||!overlap(n,lo,hi))return -1;
  if(n.left>=0){int hit=query_mesh(m,n.left,rank,lo,hi,w);return hit>=0?hit:query_mesh(m,n.right,rank,lo,hi,w);}
  for(int j=n.first;j<n.first+n.count;++j)if((!face_ranks||face_ranks[m.order[j]]<rank)&&clip(m.order[j],lo,hi,w))return m.order[j];return -1;
 }
 int query(int id,int rank,int part,const double*lo,const double*hi,double*w) const {
  const auto&n=nodes[id];if(n.minrank>=rank||!(n.mask&part)||!overlap(n,lo,hi))return -1;
  if(n.left>=0){int hit=query(n.left,rank,part,lo,hi,w);return hit>=0?hit:query(n.right,rank,part,lo,hi,w);}
  for(int j=n.first;j<n.first+n.count;++j){const auto&m=meshes[order[j]];if(m.rank<rank&&(m.part&part)){int hit=query_mesh(m,0,rank,lo,hi,w);if(hit>=0)return hit;}}return -1;
 }
};
void* create_audit(const double*v,const int64_t*f,int count,const int64_t*offsets,const int*ranks,const int*parts,int threads,const int*face_ranks){
 try{auto*a=new Audit;a->v=v;a->f=f;a->face_ranks=face_ranks;a->meshes.resize(count);a->order.resize(count);std::iota(a->order.begin(),a->order.end(),0);
  #pragma omp parallel for num_threads(threads) schedule(dynamic)
  for(int i=0;i<count;++i){auto&m=a->meshes[i];m.first=(int)offsets[i];m.count=(int)(offsets[i+1]-offsets[i]);m.rank=ranks[i];m.part=parts[i];m.order.resize(m.count);std::iota(m.order.begin(),m.order.end(),m.first);m.nodes.reserve(m.count/3+2);a->build_mesh(m,0,m.count);}
  a->build_top(0,count);return a;
 }catch(...){return nullptr;}
}
API void* layer_audit_create(const double*v,const int64_t*f,int count,const int64_t*offsets,const int*ranks,const int*parts,int threads){return create_audit(v,f,count,offsets,ranks,parts,threads,nullptr);}
API void* layer_audit_create_ordered(const double*v,const int64_t*f,int count,const int64_t*offsets,const int*ranks,const int*parts,int threads,const int*face_ranks){return create_audit(v,f,count,offsets,ranks,parts,threads,face_ranks);}
API void layer_audit_destroy(void*p){delete static_cast<Audit*>(p);}
API void layer_audit_query(void*p,int64_t count,const double*points,const int*ranks,int part,const double*lower,const double*upper,double inset,int threads,int64_t*faces,double*witnesses){
 const auto*a=static_cast<Audit*>(p);
 #pragma omp parallel for num_threads(threads) schedule(dynamic,64)
 for(int64_t i=0;i<count;++i){double lo[3],hi[3];for(int k=0;k<3;++k){lo[k]=points[3*i+k]+lower[k]+inset;hi[k]=points[3*i+k]+upper[k]-inset;witnesses[3*i+k]=NAN;}faces[i]=a->query(0,ranks[i],part,lo,hi,witnesses+3*i);}
}
