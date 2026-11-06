# PyMOL Visualization Script for 4BVX Chain B
# Color scheme matching SGAP-PPIS Figure 3:
#   Orange: True Positive (TP)
#   Green: False Negative (FN)
#   Blue: False Positive (FP)
#   Gray: True Negative (TN)

fetch 4bvx, type=pdb, async=0
select target_chain, chain B and 4bvx
hide everything, all
show surface, target_chain
set transparency, 0.1, target_chain
color gray80, target_chain

select tp_res, chain B and resi 30+32+42+43+44+45+46+47+48+51+52+55+63+67+68+70+71+72+74+75+76+78+79+81+82+83
color orange, tp_res
select fp_res, chain B and resi 1+2+3+4+5+6+7+8+9+10
color marine, fp_res
select fn_res, chain B and resi 92+95+98+99+102+103+107+111
color forest, fn_res

zoom target_chain
set ray_shadows, 0
set surface_quality, 1
# Site-level statistics: TP=26, TN=126, FP=10, FN=8