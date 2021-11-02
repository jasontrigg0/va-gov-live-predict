import csv
import pandas as pd
import random
import json
import re
import datetime

import sys
import os
sys.path.append(os.path.dirname(__file__)) #allow import from current directory
from lasso import StandardLasso

#TODO: create github repo
#TODO: setup auto-committing updates
#TODO: add more specific early voting information from VPAP
#TODO: add county demographic info
#TODO: resolve missing precincts (eg new in 2021)
#TODO: simulate real 2021 data and confirm it works
#TODO: historical csv? assumign that lags the json so skipping

def compute_features():
    pass

def diff_locality():
    df = pd.read_csv("2021_from_json.csv")
    df = df.rename(columns={"PoliticalParty": "party", "LocalityName": "county", "PrecinctName": "precinct"})
    #df["LocalityName"] = df["LocalityName"].str.upper()
    #df["PrecinctName"] = df["PrecinctName"].apply(lambda x: x.rsplit(" ",1)[0])
    l1 = set(list(df[["county","precinct"]].itertuples(index=False, name=None)))

    df = pd.read_csv("2020_from_json.csv")
    df = df.rename(columns={"PoliticalParty": "party"})
    df["precinct"] = df["precinct"].apply(map_2020_precincts)
    #df["LocalityName"] = df["LocalityName"].str.upper()
    #df["PrecinctName"] = df["PrecinctName"].apply(lambda x: x.rsplit(" ",1)[0])
    l2 = set(list(df[["county","precinct"]].itertuples(index=False, name=None)))

    # df = pd.read_csv("2016 November General.csv")
    # df = df[df["OfficeTitle"] == "President and Vice President"]
    # df = df.rename(columns={"TOTAL_VOTES": "Votes"})
    # l2 = set(df["PrecinctName"])
    print(l1)
    print(l2)
    print("----")
    print("----")
    print("----")
    print("----")
    print("----")
    # for x in sorted(l1.difference(l2)):
    #     print(x)
    for x in sorted(l2.difference(l1)):
        print(x)

def load_election_live(filename, year, simulate=False):
    #load election info from the live feed
    df = pd.read_csv(filename)
    df = df.rename(columns={"PoliticalParty": "party", "LocalityName": "county", "PrecinctName": "precinct", "Votes": "votes"})
    df["state"] = "Virginia"
    df["county"] = df["county"].apply(lambda x: re.sub("( City| County)$","",x).replace("King & Queen","King and Queen")) #rename counties to match the geojson file

    if year == 2020:
        df["precinct"] = df["precinct"].apply(map_2020_precincts)
    elif year == 2021:
        pass
    else:
        raise
    df["precinct"] = df["precinct"].apply(lambda x: re.sub(" \(\d{2}\)$","",x))

    df = df[["party","state","county","precinct","votes"]]
    df = tally_votes(df)

    #VA records its absentee voting by (county,congressional district)
    #there are 95 counties and only 11 districts so this generally means one per county
    #but a few counties touch multiple districts -> multiple rows
    #for simplicity want to have one absentee row per county
    #fortunately the multiples are merged into a single row per county by tally_votes above
    #also some counties are missing absentee rows entirely (why? is this an error?): add these if necessary

    #add missing absentee rows
    if year == 2020:
        county_wide_precincts = ["# AB - Central Absentee Precinct","## Provisional"]
    elif year == 2021:
        county_wide_precincts = ["# AB - Central Absentee Precinct","# EV - Central Absentee Precinct","# PE - Central Absentee Precinct","## Provisional"]
    else:
        raise
    to_add = []
    for key, subframe in df.groupby(["state","county"]):
        for p in county_wide_precincts:
            if not len(subframe[subframe["precinct"] == p]):
                df = df.append({"state": key[0], "county": key[1], "precinct": p, "dem": 0, "rep": 0, "other": 0}, ignore_index=True)

    #simulate a currently in progress election for testing
    if simulate == True:
        def simulate_row(x):
            if random.random() < 0.8:
                x["rep"] = 0
                x["dem"] = 0
                x["other"] = 0
            return x
        df = df.apply(simulate_row, axis=1)

    category_info = get_category_info(year)
    return process_categories(df, category_info)

def process_categories(df, category_info):
    output = {}
    for category in category_info:
        info = category_info[category]
        output[category] = {
            "level": info["level"]
        }
        def filter_fn(x):
            include_clause = (x["precinct"] in info["include"] if info.get("include") else True)
            exclude_clause = (x["precinct"] not in info.get("exclude",[]))
            return include_clause and exclude_clause
        output[category]["df"] = df[df.apply(filter_fn, axis=1)]
        if info["level"] == "precinct":
            pass
        elif info["level"] == "county":
            output[category]["df"].drop("precinct", axis=1)
        else:
            raise
        if info.get("merge"):
            if info["level"] != "county": raise #only county-level supported
            output[category]["df"] = output[category]["df"].groupby(["state","county"]).agg(sum).reset_index()
            output[category]["df"]["precinct"] = info.get("merge")
    return output


def get_category_info(year):
    return {
        2020: {
            "Election Day Votes": {
                "level": "precinct",
                "exclude": [
                    "# AB - Central Absentee Precinct",
                    "## Provisional"
                ]
            },
            "Provisional Votes": {
                "level": "county",
                "include": ["## Provisional"]
            },
            "Absentee Votes": {
                "level": "county",
                "include": ["# AB - Central Absentee Precinct"]
            }
        },
        2021: {
            "Election Day Votes": {
                "level": "precinct",
                "exclude": [
                    "# AB - Central Absentee Precinct",
                    "# EV - Central Absentee Precinct",
                    "# PE - Central Absentee Precinct",
                    "## Provisional"
                ]
            },
            "Provisional Votes": {
                "level": "county",
                "include": ["## Provisional"]
            },
            "Absentee by Mail Votes": {
                "level": "county",
                "include": [
                    "# AB - Central Absentee Precinct",
                    "# PE - Central Absentee Precinct"
                ],
                "merge": "# AB - Central Absentee Precinct"
            },
            "Advanced Voting Votes": {
                "level": "county",
                "include": ["# EV - Central Absentee Precinct"]
            }
        }
    }[year]

def merge_baselines(categories_live, categories_baseline):
    #merge baseline information into the live dataframes
    live_to_baseline = {
        "Absentee by Mail Votes": "Absentee Votes",
        "Advanced Voting Votes": "Absentee Votes"
    }
    for category in categories_live:
        baseline_category = live_to_baseline.get(category,category)
        assert(categories_live[category]["level"] == categories_baseline[baseline_category]["level"])
        level = categories_live[category]["level"]
        if level == "precinct":
            join_cols = ["county","precinct"]
        elif level == "county":
            join_cols = ["county"]
        else:
            raise
        df_baseline = categories_baseline[baseline_category]["df"]
        df_baseline = df_baseline.rename(columns={x: "baseline_"+x for x in df_baseline.columns if x not in join_cols})

        df_live = categories_live[category]["df"]
        df_live = df_live.merge(df_baseline)

        #compute baseline aggregations based on reporting precincts only
        df_live["reporting_baseline_margin"] = 1 * (df_live["total"]>0) * df_live["baseline_margin"]
        df_live["reporting_baseline_total"] = 1 * (df_live["total"]>0) * df_live["baseline_total"]

        if categories_live[category]["level"] == "precinct":
            df_live["reporting_baseline_county_margin"] = df_live.groupby("county")["reporting_baseline_margin"].transform(sum)
            df_live["reporting_baseline_county_total"] = df_live.groupby("county")["reporting_baseline_total"].transform(sum)
            df_live["reporting_baseline_county_margin"] -= df_live["reporting_baseline_margin"]
            df_live["reporting_baseline_county_total"] -= df_live["reporting_baseline_total"]

            PSEUDOCOUNTS = 10
            df_live[f"reporting_baseline_county_margin_frac"] = (df_live[f"reporting_baseline_county_margin"] + PSEUDOCOUNTS) / (df_live[f"reporting_baseline_county_total"] + PSEUDOCOUNTS)

        df_live["reporting_baseline_state_margin"] = df_live.groupby("state")["reporting_baseline_margin"].transform(sum)
        df_live["reporting_baseline_state_total"] = df_live.groupby("state")["reporting_baseline_total"].transform(sum)
        df_live["reporting_baseline_state_margin"] -= df_live["reporting_baseline_margin"]
        df_live["reporting_baseline_state_total"] -= df_live["reporting_baseline_total"]

        PSEUDOCOUNTS = 10
        df_live[f"reporting_baseline_state_margin_frac"] = (df_live[f"reporting_baseline_state_margin"] + PSEUDOCOUNTS) / (df_live[f"reporting_baseline_state_total"] + PSEUDOCOUNTS)

        categories_live[category]["df"] = df_live

def load_election_hist(filename, office):
    #TODO: finish this
    df = pd.read_csv(filename)
    df = df[df["OfficeTitle"] == office]
    df = df.rename(columns={"TOTAL_VOTES": "Votes"})
    return tally_votes(df)

def map_2020_precincts(name):
    #map from 2020 precinct names to 2021
    precinct_mapping = {
        "201 - MAURY SCHOOL (08)": "201 - NAOMI L. BROOKS SCHOOL (08)", #school renamed
        "201 - Armstrong (11)": "201 - ARMSTRONG (11)",
        "630- ARMY (08)": "630 - ARMY (08)",
        "502 - BLAIR ROAD (07)": "502 - BLAIR (07)",
        "102 - EPES (03)": "102 - STONEY RUN (03)",
        "108 - LEE HALL (03)": "108 - KATHERINE JOHNSON (03)",
        "210 - NELSON (03)": "210 - KNOLLWOOD MEADOWS (03)",
        "204 - WELLESLEY (03)": "204 - MARINERS' MUSEUM (03)",
        "011 - Mt. Hermon Village (03)": "011 - JOSEPH E. PARKER RECREATION CENTER (03)",
        "025 - JOHN TYLER ELEMENTARY SCHOOL  (03)": "025 - WATERVIEW ELEMENTARY SCHOOL (03)",
        "029 - WOODROW WILSON HIGH SCHOOL (03)": "029 - MANOR HIGH SCHOOL (03)",
        "036 - SCOTTISH RITE TEMPLE (03)": "036 - CHURCHLAND HIGH SCHOOL (03)",
        "203 - NI RIVER - ELYS FORD (07)": "203 - NI RIVER ELYS FORD (07)"
    }

    #TODO: handle merged or split precincts
    #("303 - RIDGE (09)"): ("303 - RIDGE (09)", "304 - LONGS FORK (09)")
    #("419 - LANE (08)"): ("419 - LANE #1 (08)", "430 - LANE #2 (08)")
    #("501 - BAILEYS (08)"): ("501 - BAILEYS #1 (08)", "531 - BAILEYS #2 (08)")
    #("516 - WEYANOKE (08)"): ("516 - WEYANOKE #1 (08)", "532 - WEYANOKE #2 (08)")
    #("522 - CAMELOT (11)"): ("522 - CAMELOT #1 (11)", "534 - CAMELOT #2 (11)")
    #("002 - NORTH (09)", "003 - SOUTH (09)"): ("005 - WEST (09)")
    #("301 - DENDRON (04)", "302 - WALLS BRIDGE (04)"): ("301 - DENDRON (04)")
    return precinct_mapping.get(name, name)

def tally_votes(df):
    cols = ["state","county","precinct"]
    rep_votes = df[df["party"] == "Republican"].groupby(cols)["votes"].sum().reset_index().rename(columns={"votes": "rep"})
    dem_votes = df[df["party"] == "Democratic"].groupby(cols)["votes"].sum().reset_index().rename(columns={"votes": "dem"})
    other_votes = df[(df["party"] != "Republican") & (df["party"] != "Democratic")].groupby(cols)["votes"].sum().reset_index().rename(columns={"votes": "other"})
    all_votes = rep_votes.merge(dem_votes,on=cols).merge(other_votes,on=cols)
    assert(len(all_votes) == max(len(rep_votes),len(dem_votes),len(other_votes)))

    return all_votes

def compute_stats(category_info):
    for category in category_info:
        info = category_info[category]
        info["df"] = compute_basic_stats(info["df"])
        info["df"] = add_agg_stats(info["df"], "state")
        if info["level"] == "precinct":
            info["df"] = add_agg_stats(info["df"], "county")
    return category_info

def compute_basic_stats(df):
    #ie stats that are relevant for both county and precinct-level stats
    df["margin"] = df["rep"] - df["dem"]
    df["total"] = df["rep"] + df["dem"] #ignoring third party turnout for now -- TODO: include expectations about how third-party voting changed from baseline if we have them (and/or add support for more parties???)
    df["margin_frac"] = (df["margin"] + 1e-6) / (df["total"] + 1e-6)
    df["two_party_frac"] = (df["rep"] + df["dem"]+1e-6) / (df["total"]+1e-6)
    return df

def add_agg_stats(df, level):
    df[f"{level}_margin"] = df.groupby(level)["margin"].transform(sum)
    df[f"{level}_total"] = df.groupby(level)["total"].transform(sum)
    df[f"{level}_margin"] = df.apply(lambda x: x[f"{level}_margin"] -  x["margin"] if x["total"] else x[f"{level}_margin"], axis=1) #TODO: do you need the if here?
    df[f"{level}_total"] = df.apply(lambda x: x[f"{level}_total"] -  x["total"] if x["total"] else x[f"{level}_total"], axis=1) #TODO: do you need the if statement here?

    PSEUDOCOUNTS = 10
    df[f"{level}_margin_frac"] = (df[f"{level}_margin"] + PSEUDOCOUNTS) / (df[f"{level}_total"] + PSEUDOCOUNTS)
    return df

def add_predictors(categories_live, settings):
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
    training = df[df["total"]>0]

    #first print results of a simple model meant to be easy to interpret
    simple_cols = ["baseline_margin_frac"]
    simple_margin_mdl = StandardLasso(alpha=25, prior=[1], intercept_prior=0, cols=simple_cols)
    print("---")
    print("simple margin model:")
    simple_margin_mdl.fit(training, training["margin_frac"], sample_weight=training["baseline_total"], print_fit=True)

    #full model:
    margin_mdl = get_margin_model(category, settings)

    print("---")
    print("full margin model:")
    margin_mdl.fit(training, training["margin_frac"], sample_weight=training["baseline_total"], print_fit=True)

    return margin_mdl.predict(df)

def get_turnout_model(category, settings):
    if category == "Election Day Votes":
        cnt_mdl_fields = ["est_total","est_margin"] #,"bmf_X_est","white_X_est","edu_X_est","density_X_est"]
        cnt_mdl_priors = [settings["election_day_ratio"],0] #,0,0,0,0]
    elif category == "Absentee by Mail Votes":
        cnt_mdl_fields = ["est_total","est_margin"] #,"bmf_X_est","white_X_est","edu_X_est","density_X_est"]
        cnt_mdl_priors = [settings["absentee_mail_ratio"],0] #,0,0,0,0]
    elif category == "Advanced Voting Votes":
        cnt_mdl_fields = ["est_total","est_margin"] #,"bmf_X_est","white_X_est","edu_X_est","density_X_est"]
        cnt_mdl_priors = [settings["early_voting_ratio"],0] #,0,0,0,0]
    else:
        cnt_mdl_fields = ["est_total","est_margin"] #,"bmf_X_est","white_X_est","edu_X_est","density_X_est"]
        cnt_mdl_priors = [1,0] #,0,0,0,0]

    cnt_mdl = StandardLasso(alpha = 25, fit_intercept=False, prior = cnt_mdl_priors, cols=cnt_mdl_fields)

    return cnt_mdl

def fit_predict_turnout_model(df, category, settings):
    training = df[df["total"]>0]
    #first print results of a simple model meant to be easy to interpret
    simple_turnout_mdl = StandardLasso(alpha = 25, prior=[1], fit_intercept=False)
    early_categories = ['Absentee by Mail Votes','Advanced Voting Votes']
    print("---")
    print("simple turnout model:")
    simple_turnout_mdl.fit(training[["est_total"]], training["total"], print_fit=True)

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

def generate_predictions(categories_live, settings):
    #NOTE: at first tried fitting just the absolute margin instead of the percent margin
    #which is appealing because it's the exact value you want to know about the election.
    #However, it sort of combines predicting turnout with predicting percent margin
    #and because there's no clear way to know when precincts are done reporting
    #you end up with a lot of incomplete precincts included in the regression
    #which dampens the margin projection.
    #now thinking it's better to project margin frac and turnout separately
    #for those precincts that are partially complete but whose turnout we know
    for category in categories_live:
        print("---")
        print(f"predicting {category}")
        df = categories_live[category]["df"]
        if len(df[df['total'] > 0]) == 0 or category == "Provisional Votes":
            predict_from_priors(df, category, settings)
        else:
            predict_from_model(df, category, settings)

        #generate projections from raw regression predictions
        def gen_proj(r):
            pred_frac = (r["total"]+1e-6)/(r["pred_total"]+1e-6)
            if r.get("early_total"):
                #TODO: allow for vote totals that exceed the early_total projection
                #eg in the jan 5 election there were a few percent more in-person early voting
                #than expected, while mail-in votes were about as expected (as measured on jan 23)?
                proj_total = max(r["early_total"],r["total"])
            elif category == "Election Day Votes" and pred_frac > 0.5:
                #99% of precincts go straight from 0 to fully counted for election day votes
                #so it's a safe bet every vote has been counted
                proj_total = r["total"]
            elif pred_frac > 0.5:
                #shouldn't get here, as mail-in and early votes are covered by r["early_total"]
                #precincts count in multiple steps 20% of the time for early votes and 50% of the time for mail-ins
                proj_total = r["total"]
            else:
                proj_total = max(r["pred_total"],r["total"])

            proj_frac = (r["total"]+1e-6)/(proj_total+1e-6)

            margin = r["rep"] - r["dem"]
            proj_margin = margin + r["pred_margin_frac"] * (proj_total - r["total"])
            proj_rep = round((proj_total + proj_margin) / 2)
            proj_dem = round((proj_total - proj_margin) / 2)

            return pd.Series([proj_total, proj_rep, proj_dem])

        df[["proj_total","proj_rep","proj_dem"]] = df.apply(gen_proj,axis=1)
        df["outs_total"] = df["proj_total"] - df["total"]
        df["outs_rep"] = df["proj_rep"] - df["rep"]
        df["outs_dem"] = df["proj_dem"] - df["dem"]

def write_categories(categories_live):
    #generate a county/state level file and a precinct level file
    #the website will load the high level info quickly
    #then pull the precinct level asynchronously for later drilldowns
    election = "va_gov"
    high_level_data = {}
    precinct_level_data = {}

    #high_level_data.setdefatulelection]["state"]

    export_fields = ["rep","dem","total","outs_rep","outs_dem","outs_total","proj_rep","proj_dem","proj_total"]

    for category in categories_live:
        df = categories_live[category]["df"]
        state_vals = df[export_fields].sum().to_dict()
        high_level_data.setdefault(election,{}).setdefault("state",{})[category] = state_vals
        for county, group in df.groupby(["county"]):
            county_vals = group[export_fields].sum().to_dict()
            high_level_data.setdefault(election,{}).setdefault("county",{}).setdefault(county,{})[category] = county_vals
        if categories_live[category]["level"] == "precinct":
            for _,r in df.iterrows():
                county_precinct = f"{r['county']}|{r['precinct']}"
                precinct_level_data.setdefault(election,{}).setdefault(county_precinct,{})[category] = {k:r[k] for k in export_fields}

    #add an extra "total" category which is the sum across all categories:
    for f in export_fields:
        state_info = high_level_data[election]["state"]
        state_total = sum(state_info[category][f] for category in state_info if category != "total")
        state_info.setdefault("total",{})[f] = state_total

        for county in high_level_data[election]["county"]:
            county_info = high_level_data[election]["county"][county]
            county_total = sum(county_info[category][f] for category in county_info if category != "total")
            county_info.setdefault("total",{})[f] = county_total
        for precinct in precinct_level_data[election]:
            precinct_info = precinct_level_data[election][precinct]
            precinct_total = sum(precinct_info[category][f] for category in precinct_info if category != "total")
            precinct_info.setdefault("total",{})[f] = precinct_total

    high_level_data["time"] = datetime.datetime.now().strftime("%-I:%M %p ET, %B %-d, %Y")
    with open("pred.json","w") as f_out:
        f_out.write(json.dumps(high_level_data))
    with open("precinct-pred.json","w") as f_out:
        f_out.write(json.dumps(precinct_level_data))


if __name__ == "__main__":
    #diff_locality() #check on precinct changes between 2020 -> 2021
    categories_baseline = load_election_live("2020_from_json.csv",2020)
    #load_election_hist("2016 November General.csv","President and Vice President")
    compute_stats(categories_baseline)

    #categories_live = load_election_live("2020_from_json.csv", 2020, True) #simulate from 2020 data
    categories_live = load_election_live("2021_from_json.csv",2021)
    compute_stats(categories_live)

    merge_baselines(categories_live, categories_baseline)


    #baseline settings for testing
    # settings = {
    #     "election_day_ratio": 1,
    #     "absentee_mail_ratio": 0.362,
    #     "early_voting_ratio": 0.638,
    #     "election_day_margin_diff": 0,
    #     "absentee_margin_diff": 0,
    # }


    #spitballing from VPAP early voting data
    #https://www.vpap.org/elections/early-voting/november-2021-election/
    #2021 had 858571 in person vs 1796973 in 2020
    #2021 has 283584 mail-in + ballpark 27000 outstanding = 310584 versus 1020821 in 2020
    #NOTE: the 2020 baseline includes all absentee votes together
    #whereas they're broken out in 2021
    #so roughly: settings["early_voting_ratio"] = expected_early_voting_2021 / early_plus_mail_2020
    #            settings["absentee_mail_ratio"] = expected_mail_voting_2021 / early_plus_mail_2020

    #Based on the voter file Nate Cohn's wild guess has the early vote as 62% dem
    #https://twitter.com/Nate_Cohn/status/1454505023388454919
    #vs 65.7% in the baseline 2020 presidential
    # -> 7% margin swing

    #in georgia runoff dems were +32.5% in mail-in
    #vs +1.2% in early voting
    #if we have a similar breakdown

    settings = {
        "election_day_ratio": 0.75,
        "absentee_mail_ratio": 310584 / (1796973 + 1020821),
        "early_voting_ratio": 858571 / (1796973 + 1020821),
        "election_day_margin_diff": 0,
        "early_voting_diff": 0.12 - 0.15,
        "absentee_mail_diff": 0.12 + 0.15,
    }

    add_predictors(categories_live,{})

    generate_predictions(categories_live, settings)

    write_categories(categories_live)
