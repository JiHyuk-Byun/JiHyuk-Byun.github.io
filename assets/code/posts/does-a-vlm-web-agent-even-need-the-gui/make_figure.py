import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Final 10-site results (Qwen3.5-9B, 220 tasks). (text_success, text_n, vision_success, vision_n)
data = {
 "gcalendar":(17,20,4,20),"airbnb":(13,20,1,20),"stackoverflow":(18,20,7,20),
 "linkedin":(15,20,5,20),"github":(15,20,6,20),"amazon":(11,20,3,20),
 "gmail":(33,40,20,40),"slack":(12,20,6,20),"zillow":(8,20,4,20),"ubereats":(7,20,5,20),
}
rows=[]
for s,(tp,tn,vp,vn) in data.items():
    st=tp/tn*100; sv=vp/vn*100; rows.append((s,st,sv,st-sv))
rows.sort(key=lambda r:r[3])  # ascending tax -> biggest on top after barh
sites=[r[0] for r in rows]; txt=[r[1] for r in rows]; vis=[r[2] for r in rows]; tax=[r[3] for r in rows]

TEAL="#0c8f83"; AMBER="#c9741a"
y=np.arange(len(sites)); h=0.38
fig,ax=plt.subplots(figsize=(7,5.2),dpi=130)
ax.barh(y+h/2, txt, height=h, color=TEAL, label="text-MDP (dispatch)")
ax.barh(y-h/2, vis, height=h, color=AMBER, label="vision (screenshot+coords)")
for i,(t,v,tx) in enumerate(zip(txt,vis,tax)):
    ax.text(t+1.5, y[i]+h/2, f"{t:.0f}", va="center", fontsize=8, color=TEAL)
    ax.text(v+1.5, y[i]-h/2, f"{v:.0f}", va="center", fontsize=8, color=AMBER)
    ax.text(101, y[i], f"+{tx:.0f}", va="center", fontsize=8.5, fontweight="bold", color="#444")
ax.set_yticks(y); ax.set_yticklabels(sites, fontsize=9)
ax.set_xlim(0,116); ax.set_xlabel("Success rate (%)", fontsize=10)
ax.text(108, len(sites)-0.3, "tax", fontsize=8.5, fontweight="bold", color="#444", ha="center")
ax.set_title("Text-MDP vs vision on the same web tasks (Qwen3.5-9B, 10 sites)\noverall: 68% vs 28%  →  +40pp visual-understanding tax", fontsize=10.5)
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
ax.spines[["top","right"]].set_visible(False)
ax.set_axisbelow(True); ax.xaxis.grid(True, color="#eee")
plt.tight_layout()
plt.savefig("$D/figures/tax_by_site.png", bbox_inches="tight")
print("saved figures/tax_by_site.png")
