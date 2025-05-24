import sys
import os
sys.path.append(os.path.dirname(__file__)) #allow import from current directory
from lasso import StandardLasso

def get_margin_model(category, settings):
    early_categories = ['Absentee by Mail Votes','Advanced Voting Votes']
    if category == "Absentee by Mail Votes":
        margin_mdl_fields = ["baseline_margin_frac", "baseline_margin_frac_abs"] #["primary_margin_pred","baseline_margin_frac","baseline_margin_frac_abs","white_pct","edu","density"]
        margin_mdl_priors = [1,0] #[0.25,0.75,0,0,0,0]
        intercept_prior = settings["absentee_mail_diff"]
    elif category == "Advanced Voting Votes":
        margin_mdl_fields = ["baseline_margin_frac", "baseline_margin_frac_abs"] #["primary_margin_pred","baseline_margin_frac","baseline_margin_frac_abs","white_pct","edu","density"]
        margin_mdl_priors = [1,0] #[0.25,0.75,0,0,0,0]
        intercept_prior = settings["early_voting_diff"]
    else:
        margin_mdl_fields = ["baseline_margin_frac","baseline_margin_frac_sq","baseline_margin_frac_abs"] #,"white_pct","edu","density"]
        margin_mdl_priors = [1,0,0] #,0,0,0]
        intercept_prior = settings["election_day_margin_diff"]

    demean_cols = ["baseline_margin_frac_abs"] #,"white_pct","edu","density"]
    margin_mdl = StandardLasso(alpha = 25, prior=margin_mdl_priors, intercept_prior=intercept_prior, cols=margin_mdl_fields, demean_cols=demean_cols)
    return margin_mdl

def fit_predict_margin_model(df, category, settings):
    training = df[(df["total"]>0) & (df["fit"] | df["confirmed"])]

    print(f"fitting {category}")
    
    #first print results of a simple model meant to be easy to interpret
    simple_cols = ["baseline_margin_frac"]
    simple_margin_mdl = StandardLasso(alpha=25, prior=[1], intercept_prior=0, cols=simple_cols)
    print("---")
    print("simple margin model:")
    simple_margin_mdl.fit(training, training["margin_frac"], sample_weight=training["baseline_total"], print_fit=True)

    #full model:
    margin_mdl = get_margin_model(category, settings)

    #get outliers and remove
    print("---")
    print("full margin model:")
    margin_mdl.fit(training, training["margin_frac"], sample_weight=training["baseline_total"], print_fit=True)

    return margin_mdl.predict(df)

def get_turnout_model(category, settings):
    if category == "Election Day Votes":
        cnt_mdl_fields = ["baseline_total","est_margin"] #,"est_total","bmf_X_est","white_X_est","edu_X_est","density_X_est"]
        cnt_mdl_priors = [settings["election_day_ratio"],0] #,0,0,0,0]
    elif category == "Absentee by Mail Votes":
        cnt_mdl_fields = ["baseline_total","est_margin"] #,"est_total","bmf_X_est","white_X_est","edu_X_est","density_X_est"]
        cnt_mdl_priors = [settings["absentee_mail_ratio"],0] #,0,0,0,0]
    elif category == "Advanced Voting Votes":
        cnt_mdl_fields = ["baseline_total","est_margin"] #,"est_total","bmf_X_est","white_X_est","edu_X_est","density_X_est"]
        cnt_mdl_priors = [settings["early_voting_ratio"],0] #,0,0,0,0]
    else:
        cnt_mdl_fields = ["baseline_total","est_margin"] #,"est_total","bmf_X_est","white_X_est","edu_X_est","density_X_est"]
        cnt_mdl_priors = [1,0] #,0,0,0,0]

    cnt_mdl = StandardLasso(alpha = 25, fit_intercept=False, prior = cnt_mdl_priors, cols=cnt_mdl_fields)

    return cnt_mdl

def drop_outliers(df, category, settings):
    training = df[(df["total"]>0) & (df["fit"] | df["confirmed"])]

    #remove some outliers manually
    outliers_50 = (df["total"] > 50) & ((df["rep"] == 0) | (df["dem"] == 0)) & (abs(df["margin_frac"] - df["baseline_margin_frac"]) > 0.1) & (df["confirmed"] == 0)
    outliers_20 = (df["total"] > 20) & ((df["rep"] == 0) | (df["dem"] == 0)) & (abs(df["margin_frac"] - df["baseline_margin_frac"]) > 0.3) & (df["confirmed"] == 0)
    outliers = outliers_50 | outliers_20
    df["fit"] = (1-outliers) * df["fit"] #remove from fits
    if sum(outliers):
        print("precincts with only one party votes, likely incomplete")
        df[outliers].to_csv(sys.stdout)

    #remove model outliers
    cnt_mdl = get_turnout_model(category, settings)
    cnt_mdl.fit(training, training["total"])
    preds = cnt_mdl.predict(df)
    outliers = ((preds > 3 * df["total"]) | (preds < 0.33 * df["total"])) & (df["total"]>20) & (df["confirmed"] == 0) & (df["fit"] == 1)
    df["fit"] = (1-outliers) * df["fit"] #remove from fits

    if sum(outliers):
        print("outliers!")
        df[outliers].to_csv(sys.stdout)

def fit_predict_turnout_model(df, category, settings):
    training = df[(df["total"]>0) & (df["fit"] | df["confirmed"])]

    #first print results of a simple model meant to be easy to interpret
    training = df[(df["total"]>0) & (df["fit"] | df["confirmed"])]

    if category == "Election Day Votes":
        prior = settings["election_day_ratio"]
    elif category == "Absentee by Mail Votes":
        prior = settings["absentee_mail_ratio"]
    elif category == "Advanced Voting Votes":
        prior = settings["early_voting_ratio"]
    else:
        prior = 1
    simple_turnout_mdl = StandardLasso(alpha = 25, prior=[prior], fit_intercept=False)
    early_categories = ['Absentee by Mail Votes','Advanced Voting Votes']
    print("---")
    print("simple turnout model:")

    simple_turnout_mdl.fit(training[["baseline_total"]], training["total"], print_fit=True)

    #full model
    cnt_mdl = get_turnout_model(category, settings)
    print("---")
    print("full cnt model:")
    cnt_mdl.fit(training, training["total"], print_fit=True)

    return cnt_mdl.predict(df)

def predict_from_priors(df, category, settings):
    #baseline margin
    margin_mdl = get_margin_model(category, settings)
    df["pred_margin_frac"] = 0
    if margin_mdl.fit_intercept:
        df["pred_margin_frac"] += margin_mdl.intercept_prior
    for f, prior in zip(margin_mdl.cols, margin_mdl.prior):
        df["pred_margin_frac"] += df[f] * prior

    #baseline turnout
    cnt_mdl = get_turnout_model(category, settings)
    df["pred_total"] = 0
    if cnt_mdl.fit_intercept:
        df["pred_total"] += cnt_mdl.intercept_prior
    for f, prior in zip(cnt_mdl.cols, cnt_mdl.prior):
        df["pred_total"] += df[f] * prior

def predict_from_model(df, category, settings):
    df["pred_margin_frac"] = fit_predict_margin_model(df, category, settings)
    #margin_frac must be between -1 and 1
    #TODO: change to logistic?
    df["pred_margin_frac"] = df["pred_margin_frac"].clip(-1,1)

    df["pred_total"] = fit_predict_turnout_model(df, category, settings)
    df["pred_total"] = df.apply(lambda x: round(max(x["total"],x["pred_total"])), axis=1) #TODO: is this needed?

def model_votes(category_info, settings):
    for category in category_info:
        print("---")
        print(f"predicting {category}")
        df = category_info[category]["df"]

        print(len(df[True & (df["fit"] | df["confirmed"])]))
        
        if len(df[(df["total"]>0) & (df["fit"] | df["confirmed"])]) > 0:
            print("checking for outliers:")
            drop_outliers(df, category, settings)

        df.to_csv(f"/tmp/test_{category}.csv")

        if len(df[(df["total"]>0) & (df["fit"] | df["confirmed"])]) == 0 or category == "Provisional Votes":
            predict_from_priors(df, category, settings)
        else:
            predict_from_model(df, category, settings)
