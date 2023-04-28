import numpy as np
from scipy.spatial import ConvexHull,convex_hull_plot_2d
import matplotlib.pyplot as plt
import sympy as sym
from sympy import symbols, Function, Eq, solve, I, collect, expand, simplify,\
                  Derivative, init_printing, series, evaluate, Poly, Rational
                  
x,y = symbols("x y",real=True)

def SCH(f,varlst=(x,y)):
    """
    Compute support and its convex hull. If f is not a Poly then convert it
    """
    if not isinstance(f,Poly): fpoly = Poly(f,varlst)
    else: fpoly = f
    #support=np.array([[x,y] for x,y in fpoly.monoms()],dtype=np.int32)
    support=np.array([p for p in fpoly.as_dict().keys()],dtype=np.int32)
    CH=ConvexHull(support)
    #print(support[CH.vertices])
    return support,CH
    
def Normlst(CH):
    """
    Returns the list of polygon's normals
    """
    eps = 10**(-7)
    edgelst = CH.simplices
    eqnlst = CH.equations
    #midpntlst = np.array([(S[edge[0]]+S[edge[1]])/2 for edge in edgelst],dtype=np.float64)
    normlst = []
    for eq in eqnlst:
        if np.abs(eq[0]*eq[1])>eps:
            frac = Rational(*eq[:-1])
            normal = int(np.sign(eq[1]))*np.array([frac.numerator,frac.denominator],dtype=np.int32)
        #normal = normal + [eq[-1]*frac.denominator/np.abs(eq[1])]
            normlst.append(normal)
        else: 
            normal = np.array(eq[:-1],dtype=np.int32)#+[eq[-1]]
            normlst.append(normal)
    #normlst = np.array(normlst,dtype=np.int32)
    return normlst #list(zip(normlst,midpntlst))

def Normlst_old(CH):
    """
    Returns the list of polygon's normals
    """
    eps = 10**(-7)
    edgelst = CH.simplices
    eqnlst = CH.equations
    #midpntlst = np.array([(S[edge[0]]+S[edge[1]])/2 for edge in edgelst],dtype=np.float64)
    normlst = []
    for eq in eqnlst:
        if np.abs(eq[0]*eq[1])>eps:
            coordmin = np.min(np.abs(eq[:-1]))
            normlst.append(eq/coordmin)
        else: normlst.append(eq)
    normlst = np.array(normlst,dtype=np.int32)
    return normlst #list(zip(normlst,midpntlst))
    
def GetTrunc(f,CH,edgenum,varlst=(x,y),factorize=True):
    """
    Return truncated equation corresponeding to the edge with number edgenum
    """
    eps = 10**(-7)
    if not isinstance(f,Poly): fpoly = Poly(f,varlst)
    else: fpoly = f
    if edgenum >= len(CH.equations): return Null
    fdict = fpoly.as_dict()
    trunc = Poly.from_dict({p:fdict[p] for p in fdict.keys()\
                            if np.abs(np.dot(np.append(p,1),CH.equations[edgenum]))<eps},\
                           *varlst).as_expr()
    if factorize: return trunc.factor()
    else: return trunc
        
def NPimage(CH,S,ecol="k",vcol="b"):
    """
     Returns Newton polygon 
     Parameters ecol and vcol define colors of edges and vertices
    """
    def getlimits(S):
        npS = np.array(S,dtype=np.int32)
        return np.array([np.min(npS[:,0]),np.max(npS[:,0]),np.min(npS[:,1]),\
                         np.max(npS[:,1])],dtype=np.int32)
    delta = 0.4
    polylimits = getlimits(S)
    enlarge = np.array([-delta,delta,-delta,delta],dtype=np.float64)
    
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(1,1,1)
    ax.axis('square')
    ax.axis(polylimits+enlarge)
    majorx_ticks = np.arange(polylimits[0],polylimits[1]+1,1)
    majory_ticks = np.arange(polylimits[2],polylimits[3]+1,1)
    ax.set_xticks(majorx_ticks)
    ax.set_yticks(majory_ticks)
    ax.grid(which='major',color="grey",linestyle='-',lw=1,alpha=0.5)
    ax.set_xlabel("$q_1$",fontsize=16)
    ax.set_ylabel("$q_2$",fontsize=16,rotation=0)
    for i,edge in enumerate(CH.simplices):
        ax.plot(S[edge,0], S[edge,1],ecol+'-', lw=2)
#    origin=(support[edge][0,:]+support[edge][1,:])/2
#    plt.quiver(*origin,[CH_NC[i,0]],[CH_NC[i,1]],color=['b'],scale=15)
    ax.plot(S[:,0], S[:,1], vcol+'o')
    #plt.savefig("./Images/Fig1DAN.pdf",dpi=300,bbox_inches='tight')
    #plt.show()
    return ax

def AddEdgeLabel(ax,S,CH,normlst,edgenum,text,shift=0.25):
    """
    Put given text near the edge with number edgenum
    Position of the text is selected with normlst and shifted out with shift
    """
    
    midpnt = (S[CH.simplices[edgenum][0]]+S[CH.simplices[edgenum][1]])/2
    lblpos = midpnt+shift*normlst[edgenum][:-1]
    ax.text(*lblpos,text,fontsize=16)#,position=lblpos)