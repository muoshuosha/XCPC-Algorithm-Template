[TOC]

# 一些注意点


**交之前检查一下有没有傻逼错误**

1. **数据爆int（常见），数据爆ll（遇到了容易被坑死）**
2. **多组数据全局变量的初始化**
3. **数据范围有没有开够，关freopen**
4. **__gcd(a,b)注意a,b为负数的情况**
5. **注意边界范围**
6. **小样例多造，边界大样例**
7. **图论问题⼀定要注意图不连通的问题**
8. **mid ⽤ l + (r - l) / 2 可以避免溢出和负数的问题**
9. **字符串注意字符集**
10. **清零的时候全部⽤$0$~$n+1$**
11. **读入%lf代表double    %f代表float  输出统一%f**
12. **初始化尽量加在开头，确保其能运行**
13. **注意检查return break  continue 有没有跳过必要运行的代码段（比如初始化）**
14. **概率期望题考虑算贡献**
15. **方案数题考虑dp或将原状态划分成一些好求得状态**
16. **正难则反**
16. **注意$vectot <int> v(n)$的复杂度为$O(n)$，特别在循环中定义时要注意其复杂度。**
16. multiset.end返回的是最后一个元素的长度，需要改用.rbegin


##  曾经被卡的知识点

1. 主席树求$L \le a_i \le R(l \le i \le r)$的个数
2. （树上）启发式合并
3. 各种二维DP
4. 看似不是图论，需要建图跑匹配/网络流
5. 高斯消元
6. 随机
7. 神奇构造
8. 生成函数
9. 莫比乌斯反演
10. 二分图博弈/sg函数打表博弈


# 数据结构

## 普通线段树

```c++
struct seg{
    #define ls (o << 1)
	#define rs (o << 1 | 1)
    ll sum[maxn << 2], lazy[maxn << 2], lx[maxn << 2], rx[maxn << 2];
    void push_down(int o){
        if(lazy[o]){
            sum[ls] += lazy[o] * (rx[ls] - lx[ls] + 1);
            sum[rs] += lazy[o] * (rx[rs] - lx[rs] + 1);
            lazy[ls] += lazy[o];
            lazy[rs] += lazy[o];
            lazy[o] = 0;
        }
    }
    void build(int o, int l, int r){
        sum[o] = lazy[o] = 0;
        lx[o] = l; rx[o] = r;
        if(l == r) {
            sum[o] = a[l];
            return ;
        }
        int mid = l + r >> 1;
        build(ls, l, mid);
        build(rs, mid + 1, r);
        sum[o] = sum[ls] + sum[rs];
    }
    void update(int o, int l, int r, int lx, int rx, ll val){
        if(l >= lx && r <= rx){
            sum[o] += val * (r - l + 1);
            lazy[o] += val;
            return ;
        }
        int mid = l + r >> 1;
        push_down(o);
        if(lx <= mid) update(ls, l, mid, lx, rx, val);
        if(rx > mid) update(rs, mid + 1, r, lx, rx, val);
        sum[o] = sum[ls] + sum[rs];
    }
    ll query(int o, int l, int r, int lx,int rx){
        if(l >= lx && r <= rx) return sum[o];
        ll res = 0;
        int mid = l + r >> 1;
        push_down(o);
        if(lx <= mid) res += query(ls, l, mid, lx, rx);
        if(rx > mid) res += query(rs, mid + 1, r, lx, rx);
        return res;
    }
}tr;
```

## 吉司机线段树

```c++
//区间取min  维护区间和
struct node {
    int l,r;
    ll mx,mx2,sum,cnt,lazy;
}tr[maxn<<2];

void push_up(int o){
    tr[o].mx=max(tr[ls].mx,tr[rs].mx);
    tr[o].sum=tr[ls].sum+tr[rs].sum;
    if(tr[ls].mx==tr[rs].mx){
        tr[o].cnt=tr[ls].cnt+tr[rs].cnt;
        tr[o].mx2=max(tr[ls].mx2,tr[rs].mx2);
    }
    else if(tr[ls].mx>tr[rs].mx){
        tr[o].cnt=tr[ls].cnt;
        tr[o].mx2=max(tr[ls].mx2,tr[rs].mx);
    }
    else {
        tr[o].cnt=tr[rs].cnt;
        tr[o].mx2=max(tr[rs].mx2,tr[ls].mx);
    }
}
void push_down(int o){
    if(tr[o].lazy){
        if(tr[o].lazy<tr[ls].mx){
            tr[ls].sum+=tr[ls].cnt*(tr[o].lazy-tr[ls].mx);
            tr[ls].mx=tr[ls].lazy=tr[o].lazy;
        }
        if(tr[o].lazy<tr[rs].mx){
            tr[rs].sum+=tr[rs].cnt*(tr[o].lazy-tr[rs].mx);
            tr[rs].mx=tr[rs].lazy=tr[o].lazy;
        }
        tr[o].lazy=0;
    }
}
void build(int o,int l,int r){
    tr[o]={l,r,0,0,0,0,0};
    if(l==r){
        tr[o].mx=tr[o].sum=n;
        tr[o].cnt=1;
        return ;
    }
    int mid=l+r>>1;
    build(ls,l,mid);
    build(rs,mid+1,r);
    push_up(o);
}
void update(int o,int l,int r,int lx,int rx,ll val){
    if(val>=tr[o].mx) return ;
    if(l>=lx&&r<=rx&&val>tr[o].mx2){
        tr[o].sum-=tr[o].cnt*(tr[o].mx-val);
        tr[o].mx=val;tr[o].lazy=val;
        return ;
    }
    int mid=l+r>>1;
    push_down(o);
    if(mid>=lx) update(ls,l,mid,lx,rx,val);
    if(mid<rx) update(rs,mid+1,r,lx,rx,val);
    push_up(o);
}
```
## 可持久化线段树

```c++
struct node {
	int l,r,cnt;
    ll sum;
}tr[maxn*40];
int a[maxn],b[maxn],root[maxn];
int n,m,num,cnt;
int getid(int x){
	return lower_bound(b+1,b+1+len,x)-b;
}
void update(int l,int r,int &x,int y,int pos){
	tr[++cnt]=tr[y];
	tr[cnt].cnt++,tr[cnt].sum+=(ll)b[pos];x=cnt;
	if(l==r) return ;
	int mid=l+r>>1;
	if(pos<=mid) update(l,mid,tr[x].l,tr[y].l,pos);
	else update(mid+1,r,tr[x].r,tr[y].r,pos);
}
//查询区间前k大的和
ll query2(int l,int r,int x,int y,int k){
	if(l==r) return 1ll*k*b[l];
	int mid=l+r>>1;
	int tot=tr[tr[y].l].cnt-tr[tr[x].l].cnt;
	if(tot>=k) return query2(l,mid,tr[x].l,tr[y].l,k);
	else return tr[tr[y].l].sum-tr[tr[x].l].sum+query2(mid+1,r,tr[x].r,tr[y].r,k-tot);
}
//查询区间第k小   k==1查询的是区间最小
int query(int l,int r,int x,int y,int k){
	if(l==r) return b[l];
	int mid=l+r>>1;
	int tot=tr[tr[y].l].cnt-tr[tr[x].l].cnt;
	if(tot>=k) return query(l,mid,tr[x].l,tr[y].l,k);
	else return query(mid+1,r,tr[x].r,tr[y].r,k-tot);
}
void solve(){
	scanf("%d",&n);
	for(int i=1;i<=n;i++){
		scanf("%d",&a[i]);
		b[i]=a[i];
		sum[i]=sum[i-1]+a[i];
	}
	sort(b+1,b+1+n);
	len=unique(b+1,b+1+n)-b-1;
	cnt=0;
	for(int i=1;i<=n;i++)
		update(1,len,root[i],root[i-1],getid(a[i]));
	scanf("%d",&m);
	for(int i=1;i<=m;i++){
		int l,r,k;
		scanf("%d%d%d",&l,&r,&k);
		printf("%lld\n",sum[r]-sum[l-1]-query(1,len,root[l-1],root[r],r-l+1-k)+fsum(r-l+1));
	}
}
int main(){
	int T;scanf("%d",&T);while(T--)
	solve();
	return 0;
}
```

## 树状数组

```c++

template<typename T>
struct BIT {
    vector<T> tr; 
    int n;
    BIT(){}
    BIT(int n) : n(n) { 
        tr.resize(n + 1);
    }
    void init(int n) {
        this->n = n;
        tr.resize(n + 1);
    }
    int lowbit(int x) {
        return x & (-x);
    }
    void add(int x, T v) {   //单点修改
        while(x <= n) {
            tr[x] += v;
            x += lowbit(x);
        }
    }
    T query(int x) {
        T res = 0;
        while (x) {
            res += tr[x];
            x -= lowbit(x);
        }
        return res;
    }
    T query(int l, int r) {   //区间查询
       return query(r) - query(l - 1);
    }
};


//二维树状数组
ll tr[maxn][maxn];
int lowbit(int x) {
    return x & (-x);
}
void add(int x, int y, ll v) {
    while (x <= n) {
        while (y <= m) {
            tr[x][y] += v;
            y += lowbit(y);
        }
        x += lowbit(x);
    }
}
ll query(int x, int y) {
    ll res = 0;
    while (x) {
        while (y) {
            res += tr[x][y];
            y -= lowbit(y);
        }
        x -= lowbit(x);
    }
    return res;
}
```

## ST表  O(nlogn+q)  

```c++
//对于解决没有修改的区间最值问题
//一维
int a[maxn], n;
int f[maxn][20], g[maxn][20];   //f为最大,g为最小
void init(){
    for(int i = 1; i <= n; i++) {
        f[i][0] = a[i];
        g[i][0] = a[i];
    }
    for(int j = 1; j <= 20; j++){
        int tmp = 1 << j - 1;
        for(int i = 1; i + tmp <= n; i++){
            f[i][j] = max(f[i][j - 1], f[ i + tmp][j - 1]);
            g[i][j] = min(g[i][j - 1] , g[i + tmp][j - 1]);
        }
    }
}
int get_max(int l,int r){
    assert(l <= r);
    int t = 31 - __builtin_clz(r - l + 1);
    return max(f[l][t], f[r - (1 << t) + 1][t]);
}
int get_min(int l, int r) {
    assert(l <= r);
    int t = 31 - __builtin_clz(r - l + 1);
    return min(g[l][t], g[r - (1 << t) + 1][t]);
}

//二维   O(nmlognlogm+q)
int n,m;
ll f[maxn][maxn][12][12];
inline int highbit(int x) { return 31 - __builtin_clz(x); }
inline ll calc(int x, int y, int xx, int yy, int p, int q) {
    return max(
    max(f[x][y][p][q], f[xx - (1 << p) + 1][yy - (1 << q) + 1][p][q]),
    max(f[xx - (1 << p) + 1][y][p][q], f[x][yy - (1 << q) + 1][p][q])
    );
}
void init() {
    for (int x=0;x<=highbit(n) + 1;++x)
    for (int y=0;y<=highbit(m) + 1;++y)
        for (int i=0;i<= n - (1 << x) + 1;++i)
        for (int j=0;j<= m - (1 << y) + 1;++j){
            if (!x && !y) { f[i][j][x][y] = a[i][j]; continue; }
                f[i][j][x][y] = calc(i, j,i + (1 << x) - 1, j + (1 << y) - 1,
                max(x - 1, 0), max(y - 1, 0));
            }
}
inline ll get_max(int x, int y, int xx, int yy) {
    return calc(x, y, xx, yy, highbit(xx - x + 1), highbit(yy - y + 1));
}
```

## 并查集

```c++
int fi(int x){
	return (rt[x]==x)?(x):(rt[x]=fi(rt[x]));
}
void uni(int x,int y){
	int fx=fi(x),fy=fi(y);
	if(fx!=fy){
		rt[fy]=fx;
	}
}
//操作完后对每个点find一遍
```

### 带权并查集

```c++
int rt[maxn],val[maxn]; 
int fi(int x){
	if(rt[x]==x) return x;
    int fa=fi(rt[x]);
    rt[x]=fa;
    val[fa]+=val[x];
    return fa;
}
void uni(int x,int y){
    int fx=fi(x),fy=fi(y);
    if(fx!=fy){
		rt[fy]=fx;
        val[fx]+=val[fy];
	}
}
```

## 树链剖分

**注意有根树的情况dfs不能从1号节点开始**

```c++
#include <bits/stdc++.h>

using namespace std;
typedef long long ll;
typedef pair<int,int> pii;
#define ls o<<1
#define rs o<<1|1
// #define int ll
const int maxn=1e5+5;
const int inf=0x3f3f3f3f;
const int mod=1e9+7;

struct node{
	int l,r;
	ll sum,lazy;
}tr[maxn<<2];
int n,m,r,cnt;
int sz[maxn],f[maxn],dep[maxn],son[maxn],top[maxn];
//子树节点数   父节点   深度      重儿子     链的top节点
int id[maxn];
vector<int> vi[maxn];
ll v[maxn],p,vt[maxn];
//统计父节点 节点深度 子树的节点个数 重儿子节点
void dfs1(int u,int fa){
	sz[u]=1;f[u]=fa;dep[u]=dep[fa]+1;
	int maxson=-1;
	for(auto i:vi[u]){
		if(i==fa) continue;
		dfs1(i,u);
		sz[u]+=sz[i];
		if(sz[i]>maxson)
			son[u]=i,maxson=sz[i];
	}
}
//统计top节点  节点id
void dfs2(int u,int topf){
	id[u]=++cnt;
	vt[cnt]=v[u];
	top[u]=topf;
	if(!son[u]) return ;
	dfs2(son[u],topf);
	for(auto i:vi[u]){
		if(i==f[u]||i==son[u]) continue;
        dfs2(i,i);
	}
}
//以下为线段树
void push_up(int o){
	tr[o].sum=tr[ls].sum+tr[rs].sum;
}
void push_down(int o){
	if(tr[o].lazy){
		tr[ls].lazy+=tr[o].lazy;
		tr[rs].lazy+=tr[o].lazy;
		tr[ls].lazy%=p;tr[rs].lazy%=p;
		tr[ls].sum+=(tr[ls].r-tr[ls].l+1)*tr[o].lazy;
		tr[rs].sum+=(tr[rs].r-tr[rs].l+1)*tr[o].lazy;
		tr[o].lazy=0;
	}
}
void build(int o,int l,int r){
	tr[o].l=l;tr[o].r=r;
	if(l==r){
		tr[o].sum=vt[l];
		return ;
	}
	int mid=l+r>>1;
	build(ls,l,mid);
	build(rs,mid+1,r);
	push_up(o);
}
ll query(int o,int l,int r,int lx,int rx){
	if(l==lx&&r==rx){
		return tr[o].sum;
	}
	int mid=l+r>>1;
    push_down(o);
  	if(rx<=mid)
		return query(ls,l,mid,lx,rx);
	else if(lx>mid)
		return query(rs,mid+1,r,lx,rx);
	else return (query(ls,l,mid,lx,mid)+query(rs,mid+1,r,mid+1,rx))%p;
}
void update(int o,int l,int r,int lx,int rx,int v){
	if(l==lx&&r==rx){
		tr[o].sum+=(r-l+1)*v;
		tr[o].lazy+=v;
		return ;
	}
	int mid=l+r>>1;
	push_down(o);
	if(rx<=mid)
		update(ls,l,mid,lx,rx,v);
	else if(lx>mid)
		update(rs,mid+1,r,lx,rx,v);
	else {
		update(ls,l,mid,lx,mid,v);
		update(rs,mid+1,r,mid+1,rx,v);
	}
	push_up(o);
}
//以上为线段树
//链修改
void update2(int x,int y,ll v){
	while(top[x]!=top[y]){
		if(dep[top[x]]<dep[top[y]]) swap(x,y);
		update(1,1,n,id[top[x]],id[x],v);
		x=f[top[x]];
	}
	if(dep[x]>dep[y]) swap(x,y);
	update(1,1,n,id[x],id[y],v);
}
//链查询
ll query2(int x,int y){
	ll res=0;
	while(top[x]!=top[y]){
		if(dep[top[x]]<dep[top[y]]) swap(x,y);
		res+=query(1,1,n,id[top[x]],id[x]);
		res%=p;
		x=f[top[x]];
	}
	if(dep[x]>dep[y]) swap(x,y);
	res+=query(1,1,n,id[x],id[y]);
	res%=p;
	return res;
}
int main()
{
	scanf("%d%d%d%lld",&n,&m,&r,&p);
	for(int i=1;i<=n;i++)
		scanf("%lld",&v[i]);
	for(int i=1;i<n;i++){
    int u,v;
    scanf("%d%d",&u,&v);
    vi[u].push_back(v);
    vi[v].push_back(u);
	}
	dfs1(r,0);
	dfs2(r,0);
	build(1,1,n);
	for(int i=1;i<=m;i++){
		int op,x,y;
		ll z;
		scanf("%d",&op);
		if(op==1){//链修改
			scanf("%d%d%lld",&x,&y,&z);
            update2(x,y,z%p);
		}
		else if(op==2){//链查询
			scanf("%d%d",&x,&y);
			printf("%lld\n",query2(x,y));
		}
		else if(op==3){//子树修改
			scanf("%d%lld",&x,&z);
			update(1,1,n,id[x],id[x]+sz[x]-1,z%p);
		}
		else {
			scanf("%d",&x);//子树查询
			printf("%lld\n",query(1,1,n,id[x],id[x]+sz[x]-1));
		}
	}
	return 0;
}
```

## 树上启发式合并

```c++
int flag;
void dfs1(int u,int fa){  //第一遍dfs确定重儿子
	sz[u]=1;
	int maxson=-1;
	for(auto v:vi[u]){
		if(v==fa) continue;
		dfs(v,u);
		sz[u]+=sz[v];
		if(sz[v]>maxson){
			son[u]=v;
			maxson=sz[v];
		}
	}
}
void cal(int u,int fa,int val){   //计算某一结点的贡献
	for(auto v:vi[u]){
		if(v==fa||v==flag) continue;
		cal(v,u,val);
	}
	int tmp=u;
	for(int i=1;i<=20;i++){
		t[a[u]][i][tmp%2]+=val;
		tmp/=2;
	}
}
void dfs2(int u,int fa,bool keep){
	for(auto v:vi[u]){
		if(v==fa||v==son[u]) continue;
		dfs2(v,u,0);   //优先递归轻儿子
	}
	if(son[u]) dfs2(son[u],u,1),flag=son[u];
	cal(u,fa,1);
	if(!keep){   //亲儿子则删掉贡献，重儿子则不用删贡献
		cal(u,fa,-1);
		flag=0;
	}
}
```

## 莫队算法O(nsqrt(n))

```c++
//要求离线并无修改操作
//询问区间众数的出现次数
#include <bits/stdc++.h>

using namespace std;
typedef long long ll;
const int maxn=3e5+10;
const int mod=1e9+7;

struct node {
	int l,r,id;
}que[maxn];
int ans,block;
int a[maxn],cnt[maxn],t[maxn],res[maxn];
bool cmp(node a,node b){
	if(a.l/block!=b.l/block) return a.l<b.l;
	if((a.l/block)&1) return a.r<b.r;
	else return a.r>b.r;
}
void add(int x){
	t[cnt[x]]--;
	cnt[x]++;
	t[cnt[x]]++;
	ans=max(ans,cnt[x]);
}
void del(int x){
	t[cnt[x]]--;
	if(t[cnt[x]]==0&&ans==cnt[x]) ans--;
	cnt[x]--;
	t[cnt[x]]++;
}
void solve(){
	int n,q;
	scanf("%d%d",&n,&q);
	block = ceil(sqrt(n));
	for(int i=1;i<=n;i++) scanf("%d",&a[i]);
	for(int i=1;i<=q;i++){
		scanf("%d%d",&que[i].l,&que[i].r);
		que[i].id=i;
	}
	sort(que+1,que+q+1,cmp);
	int l=1,r=0;
	for(int i=1;i<=q;i++){
		while(l<que[i].l) del(a[l++]);
		while(l>que[i].l) add(a[--l]);
		while(r<que[i].r) add(a[++r]);
		while(r>que[i].r) del(a[r--]);
		int len=r-l+1;
		if(ans<=len-ans+1) 
		res[que[i].id]=1;
		else res[que[i].id]=2*ans-len;
	}
	for(int i=1;i<=q;i++) printf("%d\n",res[i]);

}
int main(){
	solve();
	return 0;
}
```



## 字典树

```c++
struct trie{
	//son[i][j]表示第i个节点的第j个儿子的编号
	int son[maxn][26];
	//该节点结尾的单词的出现次数
	int cnt[maxn];
	int tot;
	void init(){
		for(int i=0;i<=tot;i++){
			cnt[i]=0;
			for(int j=0;j<26;j++)
				son[i][j]=0;
		}
		tot=0;
	}
	void insert(char *s){
		int len=strlen(s);
		int root=0;
		for(int i=0;i<len;i++){
			int id=s[i]-'a';
			if(!son[root][id]) son[root][id]=++tot;
			root=son[root][id];
		}
		cnt[root]++;
	}
	int find(char *s){
		int len=strlen(s);
		int root=0;
		for(int i=0;i<len;i++){
			int id=s[i]-'a';
			if(!son[root][id]) return 0;
			root=son[root][id];
		}
		return cnt[root];
	}
}tr;
```

## 差分

### 线性差分

```c++
int a[maxn];
int b[maxn];//差分数组
//每一个值减掉前一个值
for(int i=1;i<=n;i++){
    b[i]=a[i]-a[i-1];
}
```

### 树上差分

```c++
//点权
vector<int> vi[maxn]; //邻接表存树
int a[maxn],f[maxn];
int b[maxn];
void dfs(int u,int fa){
    f[u]=fa;
    b[u]=a[u]-a[fa];
    for(auto i:vi[u]){
        if(i==fa) continue;
        dfs(i,u);
    }
}
//如果是边权转换为点权，根的权值为0
```

## 单调堆栈（队列）O(n)

```c++
//防止卡常建议用数组模拟
//可以快速的找到a[i]左边/右边第一个大于/小于他的元素的下标
//单调栈
int a[maxn],l[maxn],r[maxn];
int stk[maxn];
for(int i=1;i<=n+1;i++){
		while(top&&a[stk[top]]>a[i]){
			r[stk[top]]=i;
			top--;
		}
    	l[i]=stk[top];
		stk[++top]=i;
}
//单调队列
//维护长度为k的子段的最大/小值
```
## 珂朵莉树

```c++
struct Node {
	int l, r, v;
	Node (int l, int r, int v) : l(l), r(r), v(v) {}
	bool operator < (const Node &rhs) const {
		return l < rhs.l;
	}
};

set<Node> odt;

set<Node>::iterator split(int x) {
  if (x > n) return odt.end();
  auto it = --odt.upper_bound(Node(x, 0, 0));
  if (it->l == x) return it;
  int l = it->l, r = it->r, v = it->v;
  odt.erase(it);
  odt.insert(Node(l, x - 1, v));
  return odt.insert(Node(x, r, v)).first;
}

void assign(int l, int r, int v) {
  auto itr = split(r + 1), itl = split(l);
  odt.erase(itl, itr);
  odt.insert(Node(l, r, v));
}

void work(int l, int r, int v, int x) {
  auto itr = split(r + 1), itl = split(l);
  for (; itl != itr; ++itl) {
	  // do things
  }
}
```



## 平衡树(treap)

```c++

const int N = 100010, INF = 1e8;

int n;
struct Node
{
    int l, r;
    int key, val;
    int cnt, size;
}tr[N];

int root, idx;

void pushup(int p)
{
    tr[p].size = tr[tr[p].l].size + tr[tr[p].r].size + tr[p].cnt;
}

int get_node(int key)
{
    tr[ ++ idx].key = key;
    tr[idx].val = rand();
    tr[idx].cnt = tr[idx].size = 1;
    return idx;
}

void zig(int &p)    // 右旋
{
    int q = tr[p].l;
    tr[p].l = tr[q].r, tr[q].r = p, p = q;
    pushup(tr[p].r), pushup(p);
}

void zag(int &p)    // 左旋
{
    int q = tr[p].r;
    tr[p].r = tr[q].l, tr[q].l = p, p = q;
    pushup(tr[p].l), pushup(p);
}

void build()
{
    get_node(-INF), get_node(INF);
    root = 1, tr[1].r = 2;
    pushup(root);

    if (tr[1].val < tr[2].val) zag(root);
}


void insert(int &p, int key)
{
    if (!p) p = get_node(key);
    else if (tr[p].key == key) tr[p].cnt ++ ;
    else if (tr[p].key > key)
    {
        insert(tr[p].l, key);
        if (tr[tr[p].l].val > tr[p].val) zig(p);
    }
    else
    {
        insert(tr[p].r, key);
        if (tr[tr[p].r].val > tr[p].val) zag(p);
    }
    pushup(p);
}

void remove(int &p, int key)
{
    if (!p) return;
    if (tr[p].key == key)
    {
        if (tr[p].cnt > 1) tr[p].cnt -- ;
        else if (tr[p].l || tr[p].r)
        {
            if (!tr[p].r || tr[tr[p].l].val > tr[tr[p].r].val)
            {
                zig(p);
                remove(tr[p].r, key);
            }
            else
            {
                zag(p);
                remove(tr[p].l, key);
            }
        }
        else p = 0;
    }
    else if (tr[p].key > key) remove(tr[p].l, key);
    else remove(tr[p].r, key);

    pushup(p);
}

int get_rank_by_key(int p, int key)    // 通过数值找排名
{
    if (!p) return 0;   // 本题中不会发生此情况
    if (tr[p].key == key) return tr[tr[p].l].size + 1;
    if (tr[p].key > key) return get_rank_by_key(tr[p].l, key);
    return tr[tr[p].l].size + tr[p].cnt + get_rank_by_key(tr[p].r, key);
}

int get_key_by_rank(int p, int rank)   // 通过排名找数值
{
    if (!p) return INF;     // 本题中不会发生此情况
    if (tr[tr[p].l].size >= rank) return get_key_by_rank(tr[p].l, rank);
    if (tr[tr[p].l].size + tr[p].cnt >= rank) return tr[p].key;
    return get_key_by_rank(tr[p].r, rank - tr[tr[p].l].size - tr[p].cnt);
}

int get_prev(int p, int key)   // 找到严格小于key的最大数
{
    if (!p) return -INF;
    if (tr[p].key >= key) return get_prev(tr[p].l, key);
    return max(tr[p].key, get_prev(tr[p].r, key));
}

int get_next(int p, int key)    // 找到严格大于key的最小数
{
    if (!p) return INF;
    if (tr[p].key <= key) return get_next(tr[p].r, key);
    return min(tr[p].key, get_next(tr[p].l, key));
}
```


# 字符串

## KMP

### next函数

```c++
int ne[maxn];
// next数组  可匹配后缀子串的最长前缀
void get_next(string s2) {
    for(int i = 1, j = 0; i < s2.size(); i++) {
        while(j && s2[i] != s2[j]) j = ne[j];
        if(s2[i] == s2[j]) j++;
        ne[i + 1] = j;
    }
}
//  判断s2在s1中出现的次数
int kmp(string s1, string s2) {
    int cnt=0;
    for(int i = 0, j = 0; i < s1.size(); i++) {
        while(j && s1[i] != s2[j]) j = ne[j];
        if(s1[i] == s2[j]) j++;
        if(j == s2.size()) cnt++;
    }
    return cnt;
}
```

### Z函数

```c++
vector<int> z_function_trivial(string s) {
  int n = (int)s.length();
  vector<int> z(n);
  for (int i = 1; i < n; ++i)
    while (i + z[i] < n && s[z[i]] == s[i + z[i]]) ++z[i];
  return z;
}
```

## AC自动机

```c++
//解决统计若干个模式串在文本串当中出现次数的问题(当模式串个数为1时其实就是KMP)   复杂度O(n)

char s[maxn];
int trie[maxn][26];
int cnt[maxn];
int fail[maxn];
int tot;

//模式串建立一颗字典树
void insert(char * s){
	int len=strlen(s+1);
	int now=0;
	for(int i=1;i<=len;i++){
		if(trie[now][s[i]-'a']) now=trie[now][s[i]-'a'];
		else {
			trie[now][s[i]-'a']=++tot;
			now=tot;
		}
	}
	cnt[now]++;
}
//fail指针指向父亲的fail节点的同字符儿子，若没有继续往前跳fail直到root
void getfail(){
	queue<int> q;
	for(int i=0;i<26;i++){
		if(trie[0][i]) {
			fail[trie[0][i]]=0;
			q.push(trie[0][i]);
		}
	}
	while(!q.empty()){
		int tmp=q.front();
		q.pop();
		for(int i=0;i<26;i++){
			if(trie[tmp][i]){
				fail[trie[tmp][i]]=trie[fail[tmp]][i];
				q.push(trie[tmp][i]);
			}
			else trie[tmp][i]=trie[fail[tmp]][i];
		}
	}
}
int query(char * s){
	int now=0,ans=0;
	int len=strlen(s+1);
	for(int i=1;i<=len;i++){
		now=trie[now][s[i]-'a'];
		for(int j=now;j&&cnt[j]!=-1;j=fail[j]){   //不能每次都跳到根节点，容易超时
			ans+=cnt[j];
			cnt[j]=-1;                            //计算有几个模式串出现过
		}
	}
	return ans;
}
void solve(){
	int n;scanf("%d",&n);
	for(int i=0;i<=tot;i++){
		cnt[i]=fail[i]=0;
		for(int j=0;j<26;j++)
		trie[i][j]=0;
	}//初始化
	tot=0;
	for(int i=1;i<=n;i++){
		scanf("%s",s+1);
		insert(s);
	}
	getfail();
	scanf("%s",s+1);
	printf("%d\n",query(s));
}
```

## 后缀自动机

- 根据反串建立SAM，后缀树的dfs序就是原串的字典序

```c++
struct SAM{
    const int N=maxn<<1;      //N为自动机节点总数
    int size,last;
    int len[maxn*2],link[maxn*2],to[maxn*2][26];
    int sz[maxn<<1];
    vector<int> vec[maxn<<1];
    void init(int Strlen,int chSize) {
        size=last=1;
        memset(to,0,sizeof(to[0]));
    }
    void ins(int c){   //每次插入一个新的字符
        int p,cur=++size;
        sz[size]=1;
        len[cur]=len[last]+1;
        for(p=last;p&&!to[p][c];p=link[p]){
            to[p][c]=cur;
        }
        if(!p) link[cur]=1;
        else {
            int q=to[p][c];
            if(len[q]==len[p]+1) link[cur]=q;
            else {
                int cl=++size;
                len[cl]=len[p]+1;
                link[cl]=link[q];
                memcpy(to[cl],to[q],sizeof(to[0]));
                while(p&&to[p][c]==q){
                    to[p][c]=cl;
                    p=link[p];
                }
                link[cur]=link[q]=cl;
            }
        }
        last=cur;
    }
    void build_tree(){
        for(int i=2;i<=size;i++){
            vec[link[i]].push_back(i);
        }
    }
    ll res=0;
    //注意先建树才能dfs
    void dfs(int u){
        for(int v:vec[u]){
            dfs(v);
            sz[u]+=sz[v];
        }
    }
    //sz[i]为以i这个节点结尾的子串的出现次数
}sam;
```



## 后缀数组

```c++
/* lcp: 最长公共前缀
** height[i]: lcp(sa[i], sa[i - 1]);
** h[i] = height[rk[i]]; h[i] >= h[i - 1] - 1;
** sa[i]: 后缀字典序第i大的起始下标
** rk[i]: 字符串从第i位开始后缀的字典序排名
** lcp(sa[i],sa[j])=min({height[i+1....j]})   (i<j)
*/

int sa[maxn],oldrk[maxn<<1],rk[maxn<<1],ht[maxn],cnt[maxn],id[maxn],px[maxn];
char s[maxn];
bool cmp(int x, int y, int w) {
  return oldrk[x] == oldrk[y] && oldrk[x + w] == oldrk[y + w];
}
void get_sa(int n) {
	int i, m = 300, p, w;   //m初始值为字符集大小   注意不能取26  因为'z'为122
    //初次排序
    for (i = 1; i <= m; ++i) cnt[i] = 0;
    for (i = 1; i <= n; ++i) ++cnt[rk[i] = s[i]];
  	for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
  	for (i = n; i >= 1; --i) sa[cnt[rk[i]]--] = i;
	for (w = 1;; w <<= 1, m = p) {  // m=p 就是优化计数排序值域
        //第二关键字排序
        for (p = 0, i = n; i > n - w; --i) id[++p] = i;
        for (i = 1; i <= n; ++i)
          if (sa[i] > w) id[++p] = sa[i] - w;
        //第一关键字排序
        for (i = 1; i <= m; ++i) cnt[i] = 0;
        for (i = 1; i <= n; ++i) ++cnt[px[i] = rk[id[i]]];
        for (i = 1; i <= m; ++i) cnt[i] += cnt[i - 1];
        for (i = n; i >= 1; --i) sa[cnt[px[i]]--] = id[i];
        for (i = 1; i <= n; ++i) oldrk[i] = rk[i];
        //附上新的排名
        for (p = 0, i = 1; i <= n; ++i)
          	 rk[sa[i]] = cmp(sa[i], sa[i - 1], w) ? p : ++p;
        if (p == n) {
          for (int i = 1; i <= n; ++i) sa[rk[i]] = i;
          break;
        }
  	}
    //get_height
	for (int i = 1, k = 0; i <= n; ++i) {
      if (k) --k;
      while (s[i + k] == s[sa[rk[i] - 1] + k]) ++k;
      ht[rk[i]] = k;  
	}
}
```

## manacher

```c++
char s1[maxn],s2[maxn];
int p[maxn];
//求一个字符串的最长回文子串
void manacher(){
	int len1=strlen(s1);
	s2[0]='@';
	for(int i=0;i<len1;i++){
		s2[2*i+1]='#';
        s2[2*i+2]=s1[i];
    }
    int len2=len1*2+2;
    s2[len2-1]='#';
    s2[len2]='$';
    p[1]=1;
    int id=0,mx=0;
    for(int i=1;i<=len2-1;i++){
        if(i<=mx)
            p[i]=min(p[2*id-i],mx-i);
        else p[i]=1;
        while(s2[i+p[i]]==s2[i-p[i]])
            p[i]++;
        if(p[i]+i>mx){
            mx=p[i]+i;
            id=i;
        }
    }
    int ans=0;
    int ma=0,mid;
    for(int i=1;i<=2*len1+1;i++){
        if(p[i]>ma){
            ma=p[i];
            mid=i;
        }
    }
}
```

## Lyndon串

```c++
char s[maxn];
//将一个字符串分解为多个lyndon串
void solve(){
	scanf("%s",s+1);
	int n=strlen(s+1);
	ll res=0;
	for(int i=1;i<=n;){
		int j=i,k=i+1;
		while(k<=n&&s[j]<=s[k]){
			if(s[j]<s[k]){
				j=i;
			}
			else {
				j++;
			}
			k++;
		}
		while(i<=j){
			i+=k-j;
		}
	}
}
int main(){
	int t;scanf("%d",&t);while(t--)
	solve();
}
```

## 字符串hash

```c++

typedef long long ll;
typedef unsigned long long ull; 
typedef pair<int,double> pii;
const int maxn=2e5+10;

//一般采用双哈希比较保险
ull base=13331;  //通常用这个作为基底，不容易产生冲突
char s[maxn];
int n;
ull b[maxn],pre[maxn];
//将字符串映射成一个ull的整数，通常利用自然溢出（有时也会对某一质数取模）
void init(int n){
    b[0] = 1;
    for(int i = 1;i < n; i++) {
    	b[i] = b[i - 1] * base;
    }
}
ll cal(int l, int r){
    return pre[r] - pre[l - 1] * b[r - l + 1];
}
void solve(){
    init();
	scanf("%d", &n);
	scanf("%s", s + 1);
    int len = strlen(s + 1);
    for(int i = 1; i <= len; i++) {
        pre[i] = pre[i - 1] * base + s[i];
    }
}
```



# 图论

## 一些结论

1. 对于一棵无向树,我们要使得其变成边强连通图,需要添加的边数== **(树中叶子节点数+1)/2**。
2. 强连通的一个最主要特征就是每个点的入度和出度都不为0，对于一个DAG，令a为树根数，b为叶子数，则答案就为max(a, b);特别的，当只有一个点时，答案为0。

## Floyd

```c++
// 注意枚举顺序
for (int k = 1; k <= n; k++)
		for (int i = 1; i <= n; i++)
			for (int j = 1; j <= n; j++)
				dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]);
```



## dij O(nlogn+m)

```c++
typedef pair<ll, int> pli;
ll dis[maxn];
bool vis[maxn];
void dij(int s) {
	priority_queue<pli, vector<pli>, greater<pli> > q;
	memset(dis, 0x3f, sizeof dis);
	memset(vis, 0, sizeof vis);
	dis[s] = 0; q.push({dis[s], s});
	while(!q.empty()) {
		int u = q.top().second; q.pop();
		if(vis[u]) continue;
		vis[u] = true;
		for(int i = 0; i < ve[u].size(); i++) {
			int v = ve[u][i].v;
			ll w = ve[u][i].val;
			if(dis[v] > dis[u] + w) {
				dis[v] = dis[u] + w;
				q.push({dis[v], v});
			}
		}
	}
}
```

## SPFA  O(n*m)

```c++
ll dis[maxn];
bool vis[maxn];
vector<pil> vi[maxn];
void spfa(int s){
    memset(dis,0x3f,sizeof(dis));
    memset(vis,0,sizeof(vis));
    queue<int> q;
    dis[s]=0;vis[s]=true;
    q.push(s);
    while(!q.empty()){
        int u=q.fornt();q.pop();
        vis[u]=false;
        for(auto i:vi[u]){
            int v=i.first;
            ll w=i.second;
            if(dis[v]>dis[u]+w){
                dis[v]=dis[u]+w;
                if(!vis[v]){
                    vis[v]=true;
                    q.push(v);
                }
            }
        }
    }
}

//求负环
//cnt记录每个点的迭代次数
bool spfa(){
	int hh=0,tt=1;
	memset(dis,0,sizeof(dis));
	memset(vis,0,sizeof(vis));
	memset(cnt,0,sizeof(cnt));
	for(int i=1;i<=n;i++)
		q[tt++]=i;
	while(hh!=tt){
		int u=q[hh++];
		vis[u]=false;
		for(auto i:vi[u]){
			int v=i.first;
			ll val=i.second;
			if(dis[v]>dis[u]+val){
				dis[v]=dis[u]+val;
				cnt[v]=cnt[u]+1;
				if(cnt[v]>n) return false;  //有负环
				if(!vis[v]){
					q[tt++]=v;
					vis[v]=true;
					cnt[v]++;
				}
			}
		}
	}
	return true;
}
```

## 网络流(Dinic)

```c++
//理论复杂度O(n^2*m)，实际远小于O(n^2*m)   比如对于二分图最大匹配  O(sqrt(n)*m)
**注意检查建图**
int n,m,s,t;
struct edge{
	int to,nxt;
	ll flow;
}e[maxn];
int head[maxn],tot,d[maxn],cur[maxn];
void addedge(int u,int v,ll c){
   e[tot].to=v;
   e[tot].nxt=head[u];
   e[tot].flow=c;
   head[u]=tot++;
   e[tot].to=u;
   e[tot].nxt=head[v];
   e[tot].flow=0;
   head[v]=tot++;
}
void init(){
	memset(head,-1,sizeof(head));
}
bool bfs(){
	for(int i=0;i<=t;i++){
		d[i]=inf;
		cur[i]=head[i];
	}
	queue<int> q;
	q.push(s);
	d[s]=0;
	while(!q.empty())
	{
		int u=q.front();
		q.pop();
		for(int i=head[u];i!=-1;i=e[i].nxt)
		{
			int v=e[i].to;
			if(e[i].flow>0&&d[v]==inf){
				q.push(v);
				d[v]=d[u]+1;
				if(v==t)
					return true;
			}
		}
	}
	return false;
}
ll dfs(int now,ll flow){
	if(t==now) return flow;
	ll res=0;
	for(int i=cur[now];i!=-1&&flow>0;i=e[i].nxt)
	{
		cur[now]=i;
		int v=e[i].to;
		if(e[i].flow>0&&(d[v]==d[now]+1)){
        ll k=dfs(v,min(flow,e[i].flow));
        e[i].flow-=k;
        e[i^1].flow+=k;
        flow-=k;
        res+=k;
        if(flow<=0) break;
		}
	}
	return res;
}
void Dinic(){
	ll max_flow=0,nowflow=0;
	while(bfs()){
	while(nowflow=dfs(s,inf)){
    max_flow+=nowflow; 	
    } 
    }
	printf("%lld\n",max_flow);
}
```

## 费用流   O(nmf)  f为最大流量

```c++
**注意检查建图**

struct Edge {
    int to, nxt, flow;
    ll cost;
} e[maxn];
int s, t, tot;
ll dis[maxn];
bool in[maxn];
int pre[maxn], a[maxn], head[maxn];
void add_edge(int u, int v, int flow, ll cost) {
    e[tot].to = v;
    e[tot].nxt = head[u];
    e[tot].flow = flow;
    e[tot].cost = cost;
    head[u] = tot++;
    e[tot].to = u;
    e[tot].nxt = head[v];
    e[tot].flow = 0;
    e[tot].cost = -cost;
    head[v] = tot++;
}
// 不要忘记init
void init() {
    memset(head, -1, sizeof(head));
    tot = 0;
}
bool spfa(int &flow, ll &cost) { //最短路对应最小费用   最长路对应最大费用
    for (int i = s; i <= t; i++) {
        dis[i] = INF;
        pre[i] = -1;
    }
    queue<int> q;
    q.push(s);
    dis[s] = 0;
    in[s] = true;
    a[s] = inf; 
    while (!q.empty()) {
        int u = q.front();
        q.pop();
        for (int i = head[u]; i != -1; i = e[i].nxt) {
            int v = e[i].to;
            ll c = e[i].cost;
            if (e[i].flow != 0 && dis[v] > dis[u] + c) {
                dis[v] = dis[u] + c;
                a[v] = min(a[u], e[i].flow);
                pre[v] = i;
                if (!in[v]) {
                    in[v] = true;
                    q.push(v);
                }
            }
        }
        in[u] = false;
    }
    if (dis[t] == INF)
        return false;
    flow += a[t];
    cost += dis[t] * a[t];
    int u = t;
    while (u != s) {
        e[pre[u]].flow -= a[t];
        e[pre[u] ^ 1].flow += a[t];
        u = e[pre[u] ^ 1].to;
    }
    return true;
}
ll MCMF() {
    int flow = 0;
    ll cost = 0;
    while(spfa(flow, cost)) ;
    return cost;
}
```



## 倍增LCA  O(qlogn)

```c++
int n;
int f[maxn][25];
int dep[maxn];
vector<vector<int>> edges;
void dfs(int u, int fa) {
	f[u][0] = fa;
	dep[u] = dep[fa] + 1;
	for(int i = 1; (1 << i) <= dep[u]; i++){
		f[u][i] = f[f[u][i - 1]][i - 1];
	}
	for(auto v : edges[u]){
		if(v == fa) continue;
		dfs(v, u);
	}
}
//u往上跳len步
int get(int u,int len) {
    for(int i = 0; i < 20; i++) {
        if((len >> i) & 1)
            u = f[u][i];
    }
    return u;
}
int lca(int x, int y) {
	if(dep[x] < dep[y]) swap(x, y);
	for(int i = 20; i >= 0; i--) {
		if(dep[f[x][i]] >= dep[y]) x = f[x][i];
		if(x == y) return x;
	}
	for(int i = 20; i >= 0; i--) {
		if(f[x][i] != f[y][i]) {
			x = f[x][i]; y = f[y][i];
		}
	}
	return f[x][0];
}
```

## tarjan LCA(离线)   O(n+q)

```c++
//1类点  当前正在dfs的点    2类点 已经dfs并回溯的点   3类点  还未dfs的点
void tarjan(int u){
  st[u]=1;
  for(auto v:vi[u]){
    if(!st[v]){
      tarjan(v);
      rt[v]=u;
    }
  }
  for(auto i:query[u]){
    int v=i.first,id=i.second;
    if(st[v]==2){
      res[id]=fi(v);
    }
  }
  st[u]=2;
}
```

## 树的重心

```c++
//最大子树的节点数最小即为重心
int sz[maxn];  //当前节点子树的节点数
int mxson[maxn];  //当前节点为根的最大子树的节点树  注意不是重儿子
vector<int> vi[maxn];
int rt;  //重心
void dfs(int u,int fa){
	sz[u]=1;
	for(auto v:vi[u]){
		if(v==fa) continue;
		dfs(v,u);
		sz[u]+=sz[v];
		mxson[u]=max(mxson[u],sz[v]);
	}
	mxson[u]=max(mxson[u],n-sz[u]);
    if(!rt||mxson[u]<mxson[rt]){
        rt=u;
    }
}
```

## 二分图匹配

- **最小点覆盖=最大匹配**
- **最大独立集=顶点数-二分图匹配数**
- **DAG最小路径覆盖数=节点数-拆点后二分图最大匹配数**

### 匈牙利O（n*m）

```c++
vector<int> vi[maxn]
int vis[maxn];
int mx[maxn];//mx[i]表示i这个点匹配的节点
int lef[maxn];
bool find(int u){
	for(int v:vi[u]){
		if(!vis[v]){
			vis[v]=1;
			if(!mx[v]||find(mx[v])){
				mx[v]=u;
				return true;
			}
		}
	}
	return false;
}
int match(){
    int res=0;
    //遍历左部的点
    for(int i=1;i<=n;i++){
        memset(vis,0,sizeof(vis));
        if(find(i)) res++;
    }
    return res;
}
```

### HK O（nsqrt(m))  

```c++
int dist[N<<1],mx[N],my[N],m,n;
vector<int> map[N];
int que[N<<1],head,tail;
int bfs(){
    int i;
    head=0;tail=-1;
    for(i=1;i<=n;i++)
        if(mx[i]==-1) que[++tail]=i;
        for(i=0;i<=m+n;i++) dist[i]=0;
        int flag=0;
        while(head<=tail){
            int u=que[head++];
            for(i=0;i<map[u].size();i++){
                int v=map[u][i];
                if(dist[n+v]==0){
                    dist[n+v]=dist[u]+1;
                    if(my[v]!=-1){
                        dist[my[v]]=dist[n+v]+1;
                        que[++tail]=my[v];
                    }
                    else flag=1;
                }
            }
        }
        return flag;
}
int dfs(int u)
{
    for(int i=0;i<map[u].size();i++)
    {
        int v=map[u][i];
        if(dist[u]+1==dist[v+n])
        {
            int t=my[v];
            dist[v+n]=0;
            if(t==-1||dfs(t))
            {
                my[v]=u;
                mx[u]=v;
                return 1;
            }
        }
    }
    return 0;
}
int H_K()
{
    int i;
    for(i=0;i<=n;i++) mx[i]=-1;
    for(i=0;i<=m;i++) my[i]=-1;
    int ans=0;
    while(bfs())
    {
        for(i=1;i<=n;i++)
            if(mx[i]==-1&&dfs(i)) ans++;
    }
    return ans;
}
```

### KM O(n^3)

```c++
struct KM{
    //默认为最大权匹配   求最小权匹配记得把权值改为相反数   对于权值是浮点数的记得把ll改为double
    const int N = 455;
    ll w[N][N]; // 边权
    ll la[N], lb[N], upd[N]; // 左、右部点的顶标
    bool va[N], vb[N]; // 访问标记：是否在交错树中
    int match[N]; // 右部点匹配了哪一个左部点
    int last[N]; // 右部点在交错树中的上一个右部点，用于倒推得到交错路
    int n;
    void init(int n,const vector<vector<int>> &a){
        this->n=n;
        for(int i=1;i<=n;i++){
            match[i] = 0;
            for(int j=1;j<=n;j++){
                w[i][j]=a[i][j];
            }
        }
    }
    bool dfs(int x, int fa) {
        va[x] = 1;
        for (int y = 1; y <= n; y++){
            if (!vb[y]){
                if (la[x] + lb[y] == w[x][y]) { // 相等子图
                    vb[y] = 1; last[y] = fa;
                    if (!match[y] || dfs(match[y], y)) {
                        match[y] = x;
                        return true;
                    }
                }
                else if (upd[y] > la[x] + lb[y] - w[x][y] ) {
                    upd[y] = la[x] + lb[y] - w[x][y];
                    last[y] = fa;
                }
            }
        }
        return false;
    }
    ll KM_val() {
        for (int i = 1; i <= n; i++) {
            la[i] = -1e18;
            lb[i] = 0;
            for (int j = 1; j <= n; j++)
                la[i] = max(la[i], w[i][j]);
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= n; j++) {
                va[j] = vb[j] = false;
                upd[j]=1e18;last[j]=0;
            }
            // 从右部点st匹配的左部点match[st]开始dfs，一开始假设有一条0-i的匹配
            int st = 0; match[0] = i;
            while (match[st]) { // 当到达一个非匹配点st时停止
                ll delta = 1e18;
                if (dfs(match[st], st)) break;
                for (int j = 1; j <= n; j++)
                    if (!vb[j] && delta > upd[j]) {
                        delta = upd[j];
                        st = j; // 下一次直接从最小边开始DFS
                    }
                for (int j = 1; j <= n; j++) { // 修改顶标
                    if (va[j]) la[j] -= delta;
                    if (vb[j]) lb[j] += delta; else upd[j] -= delta;
                }
                vb[st] = true;
            }
            while (st) { // 倒推更新增广路
                match[st] = match[last[st]];
                st = last[st];
            }
        }
        ll res=0;
        for(int i=1;i<=n;i++) {
            res+=la[i]+lb[i];
        }
        return res;
    }
}km;
```



## tarjan

### 桥边  双联通分量缩点（无向图）

```c++
struct edge{
	int to,nxt;
}e[maxn];
int cnt;
int head[maxn],tot;
//  dfs序     能回溯到前面的最小id
int dfn[maxn],low[maxn];
bool bri[maxn];

//求桥边
void tarjan(int u,int id){
	dfn[u]=low[u]=++cnt;
	for(int i=head[u];i!=-1;i=e[i].nxt){  //链式向前星存图
		int v=e[i].to;
		if(!dfn[v]){
			tarjan(v,i);
			low[u]=min(low[u],low[v]);
            if(low[v]>dfn[u]){
				ans++;
				bri[i]=bri[i^1]=true;
			}
		}
		else if((id^1)!=i){  //对于无向图加上括号里的条件
			low[u]=min(low[u],dfn[v]);
		}
	}
}
//缩点
void dfs(int u){
    f[u]=idx;
    for(int i=head[u];i!=-1;i=e[i].nxt){
        int v=e[i].to
        if(f[v]||bri[i]) continue;
        dfs(v);
    }
}
```

### 强联通分量缩点(有向图)

```c++
void tarjan(int u,int fa){
	dfn[u]=low[u]=++cnt;
	in[u]=true;
	s.push(u);
	for(auto v:vi[u]){
		if(!dfn[v]){
			tarjan(v,fa);
			low[u]=min(low[u],low[v]);
		}
		else if(in[v])low[u]=min(low[u],dfn[v]);
	}
	if(low[u]==dfn[u]){
		idx++;
		while(1){
			int v=s.top();s.pop();
			f[v]=idx;
			sz[idx]++;
			in[v]=false;
			if(v==u) break;
		}
	}
}
```

### 割点(有向/无向图通用)

```c++

bool bridge[maxn],cut[maxn];
int add_block[maxn],idx,dfn[maxn],low[maxn];
vector<int> vi[maxn];
void tarjan(int u,int fa){
    dfn[u]=low[u]=++idx;
    int son=0;
    for(int v:vi[u]){
        if(!dfn[v]){
            son++;
            tarjan(v,fa);
            low[u]=min(low[u],low[v]);
            if(u!=fa&&low[v]>=dfn[u]){
                cut[u]=true;
                add_block[u]++;
            }
        }
        else  low[u]=min(low[u],dfn[v]);
    }
    if(u==fa&&son>1) {
        cut[u]=true;
        add_block[u]=son-1;
    }
}
```

## 点分治

```c++
void get_rt(int u,int fa){  //获得新的分治节点(重心)
	sz[u]=1;mxson[u]=0;
	for(int i=head[u];i!=-1;i=e[i].nxt){
		int v=e[i].to;
		if(v==fa||vis[v])  continue;
		get_rt(v,u);
		sz[u]+=sz[v];
		mxson[u]=max(mxson[u],sz[v]);
	}
	mxson[u]=max(mxson[u],num-sz[u]);
	if(mxson[u]<mxson[rt]) rt=u;
}
void dfs(int u){
	ans+=cal(u,0);  //加上经过u节点的链的贡献
	vis[u]=true;
	for(int i=head[u];i!=-1;i=e[i].nxt){
		int v=e[i].to;
		if(vis[v]) continue;
		ans-=cal(v,e[i].len%3);  //根据容斥减掉端点在同一子树的贡献
		num=sz[v];
		rt=0; get_rt(v,0);
		dfs(rt);
	}
}
```

## 差分约束

```c++

```



# 数学

## 取模类

```c++
// assume -P <= x < 2P
int norm(int x) {
    if (x < 0) {
        x += P;
    }
    if (x >= P) {
        x -= P;
    }
    return x;
}
template<class T>
T power(T a, int b) {
    T res = 1;
    for (; b; b /= 2, a *= a) {
        if (b % 2) {
            res *= a;
        }
    }
    return res;
}
struct Z {
    int x;
    Z(int x = 0) : x(norm(x)) {}
    int val() const {
        return x;
    }
    Z operator-() const {
        return Z(norm(P - x));
    }
    Z inv() const {
        assert(x != 0);
        return power(*this, P - 2);
    }
    Z &operator*=(const Z &rhs) {
        x = i64(x) * rhs.x % P;
        return *this;
    }
    Z &operator+=(const Z &rhs) {
        x = norm(x + rhs.x);
        return *this;
    }
    Z &operator-=(const Z &rhs) {
        x = norm(x - rhs.x);
        return *this;
    }
    Z &operator/=(const Z &rhs) {
        return *this *= rhs.inv();
    }
    friend Z operator*(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res *= rhs;
        return res;
    }
    friend Z operator+(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res += rhs;
        return res;
    }
    friend Z operator-(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res -= rhs;
        return res;
    }
    friend Z operator/(const Z &lhs, const Z &rhs) {
        Z res = lhs;
        res /= rhs;
        return res;
    }
};
```

 

## 快速幂

```c++
//循环的求法
ll qpow(ll a,ll k,ll mod){
    a%=mod;
    ll res=(mod!=1)&&(a);
    while(k){
        if(k&1) res=res*a%mod;
        a=a*a%mod;
        k>>=1;
    }
    return res;
}

//递归的求法
ll qpow(ll a,ll k){
    if(k==0) return 1;
    ll res=qpow(a,k/2);
    res=res*res%mod;
    if(k&1) res=res*a%mod;
    return res;
}
```

## 龟速乘  

```c++
//计算大数的a*b%p时候不用写高精度了  但复杂度为log(n)
ll add(ll a,ll b,ll p){
    if(b==0) return 0;
    ll res=2*add(a,b/2,p)%p;
    if(b&1) res=(res+a)%p;
    return res;
}
```

## 素数测试

- 前置：快速乘、快速幂
- int 范围内只需检查 2,7,61
- long long范围内只需检查2, 325, 9375, 28178, 450775, 9780504, 1795265022
- http://miller-rabin.appspot.com/

```c++
typedef long long ll;
typedef long long LL;
ll mul(ll a,ll b,ll mod){
    a%=mod;b%=mod;
    ll res=0;
    while(b){
        if(b&1){
            res+=a;
            if(res>=mod) res-=mod;
        }
        a+=a;
        if(a>=mod) a-=mod;
        b>>=1;
    }
    return res;
}  
ll qpow(ll a,ll k,ll mod){
    ll res=mod!=1;
    a%=mod;
    while(k){
        if(k&1) res=mul(res,a,mod);
        a=mul(a,a,mod);
        k>>=1;
    }
    return res;
}
bool checkQ(LL a, LL n) {
     if (n == 2) return 1;
     if (n == 1 || !(n & 1)) return 0;
     LL d = n - 1;
     while (!(d & 1)) d >>= 1;
     LL t = qpow(a, d, n); // 不⼀定需要快速乘
     while (d != n - 1 && t != 1 && t != n - 1) {
    	 t = mul(t, t, n);
     	 d <<= 1;
     }
     return t == n - 1 || d & 1;
}
bool primeQ(LL n) {
     static vector<LL> t = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};
     if (n <= 1) return false;
     for (LL k: t) if (!checkQ(k, n)) return false;
     return true;
}
```



## 离散化

```c++
vector<int> b;
int getid(int x){
    return lower_bound(b.begin(),b.end(),x)-b.begin()+1;
}
int main(){
for(int i=1;i<=n;i++){
    int x;scanf("%d",&x);
    b.push_back(x);
}
sort(b.begin(),b.end());
b.erase(b.unique(b.begin(),b.end()),b.end());
}
```

## 求组合数

```c++
//fac存的是阶乘，ni存的是对应的逆元
ll fac[maxn],ni[maxn];
ll kpow(ll a, ll n) {
    ll res = 1;
    while(n) {
        if(n & 1) res = res * a % mod;
        n >>= 1;
        a = a * a % mod;
    }
    return res;
}
ll calc(ll n,ll m){
    if(n < m) return 0;
    return fac[n] * ni[n - m] % mod * ni[m] % mod;
}
void init(){
	fac[0] = 1;
	ni[0] = 1;
	for(ll i = 1; i < maxn; i++){
		fac[i] = fac[i - 1] * i % mod;
		ni[i] = kpow(fac[i], mod - 2);
	}
}
```

## 矩阵快速幂

```c++
struct martix{
	ll a[105][105];
}e;
ll n,k;
//矩阵相乘
martix multipy(martix A,martix B){
	martix temp;
    for(int i=1;i<=n;i++)
    for(int j=1;j<=n;j++)
    temp.a[i][j]=0;
	for(int i=1;i<=n;i++)
	for(int j=1;j<=n;j++)
	for(int k=1;k<=n;k++){
		temp.a[i][j]+=A.a[i][k]*B.a[k][j];
		temp.a[i][j]%=mod;
	}
    return temp;
}
//矩阵A的k次幂
martix pow_bound(martix A,ll k){
	if(k==0)
	return e;
	martix temp=pow_bound(A,k/2);
	temp=multipy(temp,temp);
	if(k&1)
	temp=multipy(temp,A);
	return temp;
}
```

## 线性筛（欧拉筛）

```c++
int tot,prime[maxn],v[maxn];
//v[i]存的是i这个数的最小质因子
void getPrime(ll n){
	for(int i=2;i<=n;i++){
		if(!v[i]){ 
        prime[++tot]=i;v[i]=i;
        }
		for(ll j=1;j<=tot&&prime[j]<=n/i;j++){
            v[i*prime[j]] = prime[j];
			if(i%prime[j]==0) break;   //i已经有质因子prime[j]了 所以不需要更大的了
		}
	}
}
```

## 高斯消元O(n^3)

```c++
//存0 1数据时可以用bitset优化常数   O(n^3/64)
//一般来说double够用了
const long double eps=1e-14;

long double a[100][100];   //增广矩阵
long double ans[100];   //解集
int n;
void Gauss(){
    int cnt=0;
	for(int r=1,c=1;c<=n;c++){
		//当前处理第i个变量
		int t=r;
		for(int j=r+1;j<=n;j++){
			if(fabs(a[j][c])>fabs(a[t][c]))
				t=j;
		}
		if(fabs(a[t][c])<=eps){
			//代表有自由基(没有唯一解)
            cnt++;     // 记录自由基的个数
			continue;
		}
		if(t!=r) swap(a[t],a[r]);   //交换不要忘记
		long double div=a[r][c];
		for(int j=c;j<=n+1;j++){   //将当前主元系数化成1
			a[r][j]/=div;
		}
		for(int j=r+1;j<=n;j++){
			div=a[j][c];
			for(int k=c;k<=n+1;k++){
				a[j][k]-=div*a[r][k];
			}
		}
		r++;
	}
	ans[n]=a[n][n+1];    //回代求解
	for(int i=n-1;i>=1;i--){
		ans[i]=a[i][n+1];
		for(int j=i+1;j<=n;j++){
			ans[i]-=ans[j]*a[i][j];
		}
	}
	for(int i=1;i<=n;i++){
		if(i!=n)
		printf("%.3Lf ",ans[i]);
		else printf("%.3Lf\n",ans[i]);
	}
}
```

## 生成函数

$\sum_{n\ge 1}x^n = \frac{x}{1-x}$

$\sum_{n\ge 0}x^{2n} = \frac{1}{1-x^2}$

$\sum_{n\ge 0}(n+1)x^{n} = \frac{1}{(1-x)^2}$

$\sum_{n\ge 0}\binom{m}{n}x^{n} = (1+x)^m$

$\sum_{n\ge 0}\binom{m+n}{n}x^{n} = \frac{1}{(1-x)^{m+1}}$​

## 卡特兰数

### 经典问题

1. 从(1,1)走到(n,n)不经过上半部分的方案数
2. 防止1，-1序列要求前缀和>=0的方案数

$a_n$=C(2n,n)/(n+1)=C(2n,n)-C(2n,n-1) 个相同的小球放进$m$个不同的盒子的方案数(盒子可以为空)   $C(n+m-1,n)$  

等价于$\sum_{i=1}^mx_i=n$有多少种不同的非负整数解。

## 数学公式

$$
\sum_{i=1}^ni^2=n*(n+1)*(2n+1)/6
$$

gcd(a,b)=gcd(a+b,b)=gcd(a-b,b)
$$
n的约数个数为各个质因子的的(指数+1)的乘积
$$
## 线性递推求逆元

```c++
//fac[i]为阶乘  fi[i]为阶乘的逆元  inv[i]为i的逆元
ll fac[maxn],fi[maxn],inv[maxn];
for(int i=0;i<=1;i++) fac[i]=inv[i]=fi[i]=1;
for(int i=2;i<maxn;i++){
    fac[i]=fac[i-1]*i%mod; 
    inv[i]=(mod-mod/i)*inv[mod%i]%mod;
    fi[i]=inv[i]*fi[i-1]%mod;
}

```

## 数论分块 O(sqrt(n))

```c++
int main(){
    N = read();
    // n/i和n/j相等  i为左边界 j为右边界
    for(int i = 1, j; i <= N; i = j + 1){
      j = N / (N / i);
      ans += (j - i + 1) * (N / i);
    }
    printf("%d\n", ans);
    return 0;
}
```

## 扩展欧几里得求逆元

```c++
//扩展欧几里得算法求乘法逆元
//注意x,y初始化为0，最终x的值即为a在模b下的逆元
//一般情况得到的为ax+by=gcd(a,b)的解（求逆元的时候a,b互质）
void exgcd(ll a,ll b,ll &x,ll &y){
    if(b==0){x=1;y=0;return ;}
    exgcd(b,a%b,x,y);
    ll z=x;x=y,y=z-y*(a/b);
}
```

## BSGS

处理$a^x=b$（mod p）在$O(\sqrt{p})$的时间内求解。其中$a\perp p$。方程的解$x$满足$0\leq x \leq p$。（在这里需要注意，只要$a\perp p$就行了，不要求$p$​是素数)

```c++
const int mod=1e9+7;   // 注意改模数
vector<int> ans;  //存储满足条件的x

unordered_map<ll,ll> mp;  //用hash可能会冲突  用map会多一个log复杂度
void bsgs(ll a,ll b){
    ll blo=ceil(sqrt(mod));
    ll now=b;
    mp[b]=0;
    for(int i=1;i<=blo;i++){
        now=now*a%mod;
        if(!mp.count(now))
        mp[now]=i;
    }
    now=1;
    for(int i=1;i<=blo;i++){
        now=now*a%mod;
        if(mp.count(now)){
            ll x=1ll*i*blo-mp[now];
            ans.push_back(x);
        }
    }
    mp.clear();
}
```

## 莫比乌斯反演



## 线性筛莫比乌斯函数

```c++
int flg[maxn],p[maxn],mu[maxn],tot;
void getMu(int n) {
      mu[1] = 1;
      for (int i = 2; i <= n; ++i) {
            if (!flg[i]) p[++tot] = i, mu[i] = -1;
                for (int j = 1; j <= tot && i * p[j] <= n; ++j) {
                  flg[i * p[j]] = 1;
                  if (i % p[j] == 0) {
                        mu[i * p[j]] = 0;
                        break;
                  }
                  mu[i * p[j]] = -mu[i];
            }
      }
}
```

## 杜教筛

- **在O($n^{\frac{2}{3}}$)求出一类积性函数的前缀和**

### 一般形式

- $f(i)$和$g(i)$都为积性函数
- 能够在$O(1)$的时间求$(f*g)(i)$和$g(i)$的前缀和

$S(n)=\sum_{i=1}^nf(i)$

$g(1)S(n)=\sum_{i=1}^n(f*g)(i)-\sum_{i=2}^ng(i)S(\left \lfloor \frac{n}{i} \right \rfloor)$

### 莫比乌斯函数前缀和

$S(n)=\sum_{i=1}^n\mu(i)$

$S(n)=1-\sum_{i=2}^nS(\left \lfloor \frac{n}{i} \right \rfloor )$

### 欧拉函数前缀和

$S(n)=\sum_{i=1}^n\varphi (i)$

$S(n)=\frac{1}{2}n(n+1)-\sum_{i=2}^nS(\left \lfloor \frac{n}{i} \right \rfloor )$

## 一些定理

#### 组合数公式

$C_n^m+C_n^{m+1}=C_{n+1}^{m+1}$​​          $\sum_{i=m}^{m+k}C_i^n=C_{m+k}^{n+1}-C_{m}^{n+1}$​

#### 威尔逊定理

**(p-1)!%p==p-1**   **(p-2)!%p==1**

#### 费马小定理

a和p互质 **a^(p-1)==1modp** **(以上两个p为质数)**

#### 欧拉定理

a和n互质，则**a^φ(n)==1modn**(费马小定理的加强版)

#### 卢卡斯定理  O(plogpN)

![img](https://bkimg.cdn.bcebos.com/formula/a784fd83c19921085f6e12dff816ef1d.svg)

```c++
//计算C(n,m)%p的值    p为质数

const int maxn=1e6+10;
ll fac[maxn],ni[maxn];
ll kpow(ll a, ll n) {
    ll res = 1;
    while(n) {
        if(n & 1) res = res * a % mod;
        n >>= 1;
        a = a * a % mod;
    }
    return res;
}
ll cal(ll n,ll m){
	  if(n<m) return 0;
     return fac[n]*ni[n-m]%mod*ni[m]%mod;
}
ll lucas(ll n,ll m){
    if(n<mod&&m<mod) return cal(n,m);
    else  return cal(n%mod,m%mod)*lucas(n/mod,m/mod)%mod;
}
void init(){
	fac[0]=1;
	ni[0]=1;
	for(ll i=1;i<maxn;i++){
		fac[i]=fac[i-1]*i%mod;
		ni[i]=kpow(fac[i],mod-2);
	}
}
```

#### 中国剩余定理（孙子定理）

[https://www.luogu.com.cn/problem/P1495]()

```c++
int n;ll a[maxn],b[maxn];
void exgcd(ll a,ll b,ll &x,ll &y){
    if(b==0){x=1;y=0;return ;}
    exgcd(b,a%b,x,y);
    int z=x;x=y,y=z-y*(a/b);
}
int main(){
	scanf("%d",&n);
	ll tmp=1;
	for(int i=1;i<=n;i++){
		scanf("%lld%lld",&a[i],&b[i]);
		tmp=tmp*a[i];
	}
	ll sum=0;
	for(int i=1;i<=n;i++){
		ll x=0,y=0;
		ll M=tmp/a[i];
		exgcd(M,a[i],x,y);
		x%=a[i];
		while(x<0) x+=a[i];
		sum+=b[i]*M*x;
	}
	sum%=tmp;
	printf("%lld\n",sum);
	return 0;
}
```

$x^n\equiv y(modp)$等价于$an \equiv b(modp) （1 \leq x,y,a,b \leq q-1）$ 

## 欧拉函数

```c++
//定义:就是对于一个正整数n,小于n且和n互质的正整数（包括1）的个数,记作φ(n)。
//求一个数的欧拉函数 O(sqrt(n))
int euler(int n){ //返回euler(n)   
       int res=n,a=n;  
       for(int i=2;i*i<=a;i++){  
           if(a%i==0){  
               res=res/i*(i-1);//先进行除法是为了防止中间数据的溢出   
               while(a%i==0) a/=i;  
           }  
       }  
      if(a!=1) res=res/a*(a-1);  
      return res;  
}
//求1~n的欧拉函数 O(nlogn)  埃式筛法
void init(){     
     euler[1]=1;    
     for(int i=2;i<maxn;i++)    
       euler[i]=i;    
     for(int i=2;i<maxn;i++)    
        if(euler[i]==i)    
           for(int j=i;j<maxn;j+=i)    
              euler[j]=euler[j]/i*(i-1);//先进行除法是为了防止中间数据的溢出
}
//  线性筛求欧拉函数  O(n)
bool st[maxn];
int prime[maxn],tot;
void init(){
    phi[1]=1;
    for(int i=2;i<=n;i++){
        if(!st[i]){
            phi[i]=i-1;
            st[i]=true;
            prime[++tot]=i;
        }
        for(int j=1;j<=tot&&prime[j]<=n/i;j++){
            st[i*prime[j]]=true;
            if(i%pirme[j]==0){
                phi[i*prime[j]]=phi[i]*prime[j];
                break;
            }
            phi[i*prime[j]]=phi[i]*(prime[j]-1);
        }
    }
}
```

## 多项式(抄来的)

```c++
namespace Polynomial{
    #define ADD(a,b) ((a)+=(b),(a)>=P?(a)-=P:0)
    #define SUB(a,b) ((a)-=(b),(a)<0?(a)+=P:0)
    #define MUL(a,b) (ll(a)*(b)%P)
    using Poly=vector<int>;
    constexpr int P(998244353),G(3);
    int qpow(int a,int n=P-2,int x=1){
        for(;n;n>>=1,a=MUL(a,a))
            if(n&1)x=MUL(x,a);
        return x;
    }
    template<int n>
    array<int,n> INIT(){
        array<int,n> w;
        for(int i=n>>1,x=qpow(G,(P-1)/n);i;i>>=1,x=MUL(x,x)){
            w[i]=1;
            for(int j=1;j<i;j++){
                w[i+j]=MUL(w[i+j-1],x);
            }
        }
        return w;
    }
    auto w=INIT<1<<21>();
    void DFT(Poly &f){
        int n=f.size();
        for(int k=n>>1;k;k>>=1){
            for(int i=0;i<n;i+=k<<1){
                for(int j=0;j<k;j++){
                    int y=f[i+j+k];
                    f[i+j+k]=MUL(f[i+j]-y+P,w[k+j]);
                    ADD(f[i+j],y);
                }
            }
        }
    }
    void IDFT(Poly &f){
        int n=f.size();
        for(int k=1;k<n;k<<=1){
            for(int i=0;i<n;i+=k<<1){
                for(int j=0;j<k;j++){
                    int y=MUL(f[i+j+k],w[k+j]);
                    f[i+j+k]=(f[i+j]-y+P)%P;
                    ADD(f[i+j],y);
                }
            }
        }
        for(int i=0,inv=P-(P-1)/n;i<n;i++){
            f[i]=MUL(f[i],inv);
        }
        reverse(f.begin()+1,f.end());
    }
    void DOT(Poly &f,Poly &g){
        for(int i=0;i<f.size();i++){
            f[i]=MUL(f[i],g[i]);
        }
    }
    Poly operator *(Poly f,Poly g){
        int n=f.size()+g.size()-1,k=__lg(n-1)+1,s=1<<k;
        f.resize(s);
        g.resize(s);
        DFT(f),DFT(g),DOT(f,g),IDFT(f);
        return f.resize(n),f;
    }
    Poly polyInv(Poly f){
        if(f.size()==1)return {qpow(f[0])};
        int n=f.size(),k=__lg(2*n-1)+1,s=1<<k;
        Poly g=polyInv(Poly(f.begin(),f.begin()+(n+1>>1)));
        g.resize(s);
        f.resize(s);
        DFT(f),DFT(g);
        for(int i=0;i<s;i++){
            f[i]=MUL(g[i],P-MUL(f[i],g[i])+2);
        }
        IDFT(f);
        return f.resize(n),f;
    }
    Poly deriv(Poly f){
        for(int i=1;i<f.size();i++){
            f[i-1]=MUL(i,f[i]);
        }
        f.pop_back();
        return f;
    }
    Poly integ(Poly &f,int c=0){
        int n=f.size();
        Poly g(n+1);
        g[0]=c;
        for(int i=0;i<n;i++){
            g[i+1]=MUL(f[i],qpow(i+1));
        }
        return g;
    }
    Poly polyLn(Poly f){
        int n=f.size();
        //assert(f[0]==1);
        f=polyInv(f)*deriv(f);
        f.resize(n-1);
        return integ(f);
    }
    Poly polyExp(Poly f){
        if(f.size()==1)return {1};
        //assert(f[0]==0);
        int n=f.size();
        Poly g=polyExp(Poly(f.begin(),f.begin()+(n+1>>1)));
        g.resize(n);
        Poly h=polyLn(g);
        for(int i=0;i<n;i++){
            SUB(f[i],h[i]);
        }
        ADD(f[0],1);
        return f=f*g,f.resize(n),f;
    }
}
using namespace Polynomial;
```



## FFT

```c++
//注意精度的设置
//复数板子
struct Com {
	double r,i;   //r为实部   i为虚部
	Com(double r=0,double i=0):r(r),i(i){}
}a[maxn],b[maxn];

Com operator + (const Com& A,const Com & B){
	return Com(A.r+B.r,A.i+B.i);
} 
Com operator - (const Com& A,const Com & B){
	return Com(A.r-B.r,A.i-B.i);
} 
Com operator * (const Com & A,const Com & B){
	return Com(A.r*B.r-A.i*B.i,A.r*B.i+A.i*B.r);
}
//注意n必须为2的次幂且大于两个多项式最大指数相加+1
void FFT(Com *x,int n,int p){
    for(int i=0,t=0;i<n;i++){
    	if(i>t) swap(x[i],x[t]);
        for(int j=n>>1;(t^=j)<j;j>>=1);
    }
    for(int h=2;h<=n;h<<=1){
        Com wn(cos(p*2*PI/h),sin(p*2*PI/h));
        for(int i=0;i<n;i+=h){
            Com w(1,0),u;
            for(int j=i,k=h>>1;j<i+k;j++){
                u=x[j+k]*w;
                x[j+k]=x[j]-u;
                x[j]=x[j]+u;
                w=w*wn;
            }
        }
    }
    if(p==-1)
        for(int i=0;i<=n;i++)
            x[i].r/=n;
}
void conv(Com *a,Com *b,int n){
    FFT(a,n,1);
    FFT(b,n,1);
    for(int i=0;i<=n;i++) a[i]=a[i]*b[i];
    FFT(a,n,-1);
}
```

## FWT

```c++
void FWT_or(int *a,int opt)
{
    for(int i=1;i<N;i<<=1)
        for(int p=i<<1,j=0;j<N;j+=p)
            for(int k=0;k<i;++k)
                if(opt==1)a[i+j+k]=(a[j+k]+a[i+j+k])%MOD;
                else a[i+j+k]=(a[i+j+k]+MOD-a[j+k])%MOD;
}
void FWT_and(int *a,int opt)
{
    for(int i=1;i<N;i<<=1)
        for(int p=i<<1,j=0;j<N;j+=p)
            for(int k=0;k<i;++k)
                if(opt==1)a[j+k]=(a[j+k]+a[i+j+k])%MOD;
                else a[j+k]=(a[j+k]+MOD-a[i+j+k])%MOD;
}
void FWT_xor(int *a,int opt)
{
    for(int i=1;i<N;i<<=1)
        for(int p=i<<1,j=0;j<N;j+=p)
            for(int k=0;k<i;++k)
            {
                int X=a[j+k],Y=a[i+j+k];
                a[j+k]=(X+Y)%MOD;a[i+j+k]=(X+MOD-Y)%MOD;
                if(opt==-1)a[j+k]=1ll*a[j+k]*inv2%MOD,a[i+j+k]=1ll*a[i+j+k]*inv2%MOD;
            }
}
```



# 动态规划

## 背包

```c++
// 01背包
int dp[maxn];
for(int i=1;i<=n;i++){
    for(int j=V;j>=v[i];j--){
        dp[j]=min(dp[j],dp[j-v[i]]+w[i]);
    }
}
// 完全背包
for(int i=1;i<=n;i++){
    for(int j=v[i];j<=V;j++){
        dp[j]=min(dp[j], dp[j-v[i]]+w[i]);
    }
}
```

## 数位DP

```c++
//当前枚举到的位置，当前的状态(状态可以开多个) 限制 
ll dfs(int pos,ll now,int state,bool limit){
	if(pos==0) return 1;//枚举到最后一位了，判断是否满足条件
	if(!limit&&dp[pos][state]!=-1) return dp[pos][state];
	ll temp=0;
	int up=limit?a[pos]:9;
	for(int i=0;i<=up;i++){
        //转换到下一个状态
		ll next=now*10+i;
		if(pos==1){
		int lc;
        if(i)
        	lc=lcm(i,state);
        else lc=state;
        if(next%lc==0) 
        	temp+=dfs(pos-1,next,lc,limit&&i==a[pos]);
		}
		else {
		if(i)
		temp+=dfs(pos-1,next,lcm(i,state),limit&&i==a[pos]);
	    else temp+=dfs(pos-1,next,state,limit&&i==a[pos]);
	    }
	}
	if(!limit) dp[pos][state]=temp;
	return temp;
}
ll solve(ll x){
	mem(dp,-1);
	int pos=0;
    while(x){
    	a[++pos]=x%10;
    	x/=10;
    }
    return dfs(pos,0,1,true);
}
//计数转求和的话   对每一位单独考虑对答案的贡献
pll dfs(int pos,int st,bool limit,bool lead){
    if(dp[pos][st][limit][lead].first!=-1) return dp[pos][st][limit][lead];
    if(pos==n+1) {
        if(st==((1<<m)-1)){  //题目要求的条件
            return {1,0};
        }
        else return {0,0};
    }
    pll res={0,0};
    int up=limit?a[pos]:9;
    for(int i=0;i<=up;i++){
        int nxt_st=(id[i]==-1)?st:(st|(1<<id[i]));
        if((!lead)&&i==0) nxt_st=st;
        pll tmp=dfs(pos+1,nxt_st,limit&&(i==a[pos]),lead|(i!=0));
        res.first=(res.first+tmp.first)%mod;
        res.second=(res.second+tmp.second+tmp.first*i%mod*pw[n-pos]%mod)%mod;  //求和
    }
    return dp[pos][st][limit][lead]=res;
}
```

## 区间DP

```c++
int dp[maxn][maxn];
for(int l=1;l<=n;l++){//枚举起点
    for(int len=1;len<=n;len++){//枚举长度
        if(l+len-1>n)break;
        int r=l+len-1;//终点
        for(int mid=l;mid<=r;mid++){//枚举中间点
            dp[l][r]=....;
        }
    }
}
```

## DP优化

### 决策单调优化DP

假如对于某一个$dp$方程，$dp_i$的最优转移是$dp_k$，那么称$k$为$i$的决策点

而$dp$方程满足决策单调性指的是，决策点$k$随着$i$的增大保持单调不减（二维的情况稍微复杂一点，见下面的四边形不等式推决策单调性）

### 数据结构优化DP

### 斜率优化DP

### 四边形不等式优化DP

# 计算几何

## 基础计算

```c++
//叉积
int cal1(node a,node b){
    return a.x*b.y-a.y*b.x;
}
点积
int cal2(node a,node b){
    return a.x*b.x+a.y*b.y;
}
```

## 计算多边形面积

```c++
node a[maxn];
//叉积
int cal1(node a,node b){
    return a.x*b.y-a.y*b.x;
}
int main(){
    ll ans=0;
    for(int i=1;i<=n;i++){
        ans+=cal1(i,i+1);
    }
    ans=abs(ans)/2.0;
}
```

## 判断线段相交

```c++

bool is_xiangjiao(node a,node b,node c,node d)
{
     if(max(c.x,d.x)<min(a.x,b.x)||max(a.x,b.x)<min(c.x,d.x)||max(c.y,d.y)<min(a.y,b.y)||max(a.y,b.y)<min(c.y,d.y))
        return false;
     if(((d.x-a.x)*(d.y-c.y)-(d.y-a.y)*(d.x-c.x))*((d.x-b.x)*(d.y-c.y)-(d.y-b.y)*(d.x-c.x))>0.0000000001)
        return false;
     if(((c.x-a.x)*(b.y-a.y)-(c.y-a.y)*(b.x-a.x))*((d.x-a.x)*(b.y-a.y)-(d.y-a.y)*(b.x-a.x))>0.0000000001)
        return false;
     return true;
}
```

## 三维几何

```c++
const double PI=acos(-1.0);
const double eps=1e-10;
struct P;
struct L;
int sgn(double x) {return fabs(x) < eps ? 0 : (x > 0 ? 1: -1); }
struct P{
    double  x, y, z;
    P(){}
    P(double x = 0, double y = 0, double z = 0):x(x),y(y),z(z){}
    double dist(){
        return sqrt(x * x + y * y + z * z);
    }
   	void toUnit(){
        return *this/dist();
    }
};
struct L{
    P s,t;
    L(P s,P t):s(s),t(t){}
};
P operator + (const P& a, const P& b) { return P(a.x + b.x, a.y + b.y, a.z + b.z); }
P operator - (const P& a, const P& b) { return P(a.x - b.x, a.y - b.y, a.z + b.z); }
P operator * (const P& a, double k) { return P(a.x * k, a.y * k, a.z * k); }
P operator / (const P& a, double k) { return P(a.x / k, a.y / k, a.z / k); }
P dot (const P& a, const P& b) {return P(a.x * b.x, a.y * b.y, a.z * b.z); }
P det (const P& a, const P& b) {return P(a.y * b.z - a.z * b.y, a.x * b.z - a.z * b.x, a.x * b.y - b.x * a.y); }
double dist(const P& a, P& b = P(0, 0, 0) ) {return sqrt(a.x * b.x + a.y * b.y + a.z * b.z); }

```

## 定理

### 皮克定理

适用于顶点都是整点的多边形  $S=a+b/2-1(S为面积，a为多边形内部格点数，b为多边形边上格点数)$​

![img](https://img-blog.csdnimg.cn/2021061721072959.png)

# 博弈

### 自己推不明白就暴力打一张小范围的表找规律打表时可用记忆化搜索

**能转换为后手胜的状态为先手胜，只能转换为先手胜的状态为后手胜**

## Nim博弈

**多个Nim游戏的结果为各自Nim游戏的sg值的异或和**

这类模型其实很好处理。考虑二分图的**最大匹配**，如果最大匹配**一定**包含 ![[公式]](https://www.zhihu.com/equation?tex=H) ，那么先手必胜，否则先手必败。

# 特殊技巧

## 快读快写

```c++
inline ll read() {
    ll x = 0, f = 1;
    char ch = getchar();
    while (ch < '0' || ch>'9') {
        if (ch == '-')
            f = -1;
        ch = getchar();
    }
    while (ch >= '0' && ch <= '9') {
        x = (x << 1) + (x << 3) + (ch ^ 48);
        ch = getchar();
    }
    return x * f;
}
inline void write(ll x){
    if (x == 0) {
        putchar('0');
        return;
    }
    char F[200];
    int tmp = x > 0 ? x : -x;
    if (x < 0)putchar('-');
    int cnt = 0;
    while (tmp > 0){
        F[cnt++] = tmp % 10 + '0';
        tmp /= 10;
    }
    while (cnt > 0)putchar(F[--cnt]);
}
//快读
#define gc()(is==it?it=(is=in)+fread(in,1,Q,stdin),(is==it?EOF:*is++):*is++)
const int Q=(1<<24)+1;
char in[Q],*is=in,*it=in,c;
void read(long long &n){
  for(n=0;(c=gc())<'0'||c>'9';);
  for(;c<='9'&&c>='0';c=gc())n=n*10+c-48;
}

// cin cout 关闭同步流会比scanf printf要快  但是不能够混用  包括不能使用puts
ios::sync_with_stdio(false);
cin.tie(nullptr);

// 要么使用关同步流的 cin cout  要么使用 scanf printf puts
```

## 测试程序运行时间

```c++
clock_t start = clock();
// Place your codes here...

clock_t ends = clock();
cout <<"Running Time : "<<(double)(ends - start) << "ms"<<endl;
```

## 随机

```c++
mt19937 mt(chrono::steady_clock::now().time_since_epoch().count());
 
ll rng(ll l, ll r) {
    uniform_int_distribution<ll> uni(l, r);
    return uni(mt);
}
```

## 手动扩栈

```c++
#pragma comment(linker,"/STACK:100000000,100000000")
```

## 高精度

```c++
long double a;
//long doule为128位  精度会比double高许多
scanf("%Lf",&a);
printf("%Lf\n",a);
```

## 对数

```c++
以e为底  log(n)
以2为底  log2(n)
以a为底(利用换底公式)   log(b)/log(a)
```

## 读入一行字符串

```c++
cin.getline(s,100000,'\n'); //记得把前一行的回车读掉   默认回车结束   
getline(cin,str);
substr(i,j) //从i开始读j个字符的子串
```

## 全排列

```c++
next_permutation(s,s+len);
```

## 二进制库函数

```c++
__builtin_popcount(x);  //x中1的个数。
__builtin_ctz(x);  //x末尾0的个数。x=0时结果未定义。
__builtin_clz(x);  //x前导0的个数。x=0时结果未定义。
__builtin_parity(x); //x中1的奇偶性。
```
## 套路总结

1. 一些暴力明显超时的题可以看看n比较大的时候能不能特判   n比较小用暴力
2. 期望题算贡献  
3. 对于set用二分时要用 auto it=s.lower_bound(x);
4. 要求某一参数恰好等于$k$的方案数$f_k$可以转换成$g_k-g_{k-1}$ ($g_i$代表这个参数小于/大于等于$i$的方案数)
5. 对大质数取模的计数问题一般为dp或排列组合（考虑将所求状态划分成多个可计算的子集）。

