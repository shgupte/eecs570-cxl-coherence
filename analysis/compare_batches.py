import numpy as np
a=np.load("b0.npz");b=np.load("b1.npz")
ka,ca=a["keys"],a["counts"];kb,cb=b["keys"],b["counts"]
c,ia,ib=np.intersect1d(ka,kb,assume_unique=True,return_indices=True)
print("same key set frac",len(c)/len(ka))
print("counts exactly equal frac",(ca[ia]==cb[ib]).mean(), "corr",np.corrcoef(ca[ia],cb[ib])[0,1])
print("masks equal frac",(a["mask"][ia]==b["mask"][ib]).mean())
s=ca[ia]==1; print("count==1 rows frac",s.mean(),"still count1 next",(cb[ib][s]==1).mean())
