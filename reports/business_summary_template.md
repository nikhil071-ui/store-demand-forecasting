# Business Impact Summary (Template)

**Client:** ACME Retail  
**Analyst:** YOUR NAME  
**Date:** {{DATE}}

## Objective
Forecast store-item demand and recommend inventory levels to minimize stockouts and holding cost.

## Key Findings
- Overall forecast accuracy (CV RMSE): **__**  
- Top risk categories for stockouts: **__**  
- Potential stockout reduction using safety stock policy: **__%**

## Recommendations
1. Set **lead time = 7 days** with **Z=1.65** safety factor for 95% service.
2. Replenish when **On-hand < Reorder Level** from the dashboard.
3. Review **A-class items** weekly; C-class monthly.

## Next Steps
- Integrate with purchase order system.
- Add promotions/price as exogenous features.
- Automate nightly forecast refresh.
