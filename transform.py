def add_predictors(categories_live):
    for category in categories_live:
        PSEUDOCOUNTS = 10
        category_info = categories_live[category]
        df = category_info["df"]
        df["state_ratio"] = (df["state_total"] + PSEUDOCOUNTS) / (df["reporting_baseline_state_total"] + PSEUDOCOUNTS)
        df["state_diff"] = df["state_margin_frac"] - df["reporting_baseline_state_margin_frac"]
        if category_info["level"] == "precinct":
            df["county_ratio"] = (df["county_total"] + PSEUDOCOUNTS) / (df["reporting_baseline_county_total"] + PSEUDOCOUNTS)
            df["county_diff"] = df["county_margin_frac"] - df["reporting_baseline_county_margin_frac"]
            df["diff"] = df["county_diff"] - df["state_diff"]
            df["ratio"] = df["county_ratio"] / df["state_ratio"]
        else:
            df["ratio"] = 1

        df["est_total"] = df["baseline_total"] * df["ratio"]
        df["est_margin"] = df["baseline_margin"] * df["ratio"]

        #for predicting margin
        #df["est_margin_frac"] = diff + df["baseline_margin_frac"] #this variable didn't help predictions much
        df["baseline_margin_frac_abs"] = abs(df["baseline_margin_frac"])
        df["baseline_margin_frac_sq"] = abs(df["baseline_margin_frac"]) * df["baseline_margin_frac"]
