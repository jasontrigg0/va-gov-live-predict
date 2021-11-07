import csv
import pandas as pd
import random
import json
import re
import datetime

from load import load_virginia
from model import model_votes
from transform import add_predictors

#Takeaways:
#Think I have some idea of standard practice now for election predictions.
#You can guess the bottom line margin from polling.
#Early vote counts are available and you can use the voter file to match early voters against primary
#info and estimate the early vote margin.
#Then the big unknowns are day-of turnout and margin
#Some reports come in on the day-of to give a guess of turnout
#and then you can guess day-of margin so as to make the overall margin match polling
#Nate Cohn's twitter is a good source for these

#One thing that would be cool is to fit all regressions simultaneously and enforce the prior on the total margin
#which is just some linear relationship between regression coefficients I think.

#TODO: setup auto-committing updates (ehh next time, pull from Georgia)
#TODO: add more specific early voting information from VPAP (ehh next time, pull from Georgia)
#TODO: add county demographic info (ehh next time, pull from Georgia)
#TODO: error margins?

def generate_predictions(category_info, settings):
    #NOTE: at first tried fitting just the absolute margin instead of the percent margin
    #which is appealing because it's the exact value you want to know about the election.
    #However, it sort of combines predicting turnout with predicting percent margin
    #and because there's no clear way to know when precincts are done reporting
    #you end up with a lot of incomplete precincts included in the regression
    #which dampens the margin projection.
    #now thinking it's better to project margin frac and turnout separately
    #for those precincts that are partially complete but whose turnout we know
    for category in category_info:
        df = category_info[category]["df"]
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

def write_categories(category_info):
    #generate a county/state level file and a precinct level file
    #the website will load the high level info quickly
    #then pull the precinct level asynchronously for later drilldowns
    election = "va_gov"
    high_level_data = {}
    precinct_level_data = {}

    #high_level_data.setdefatulelection]["state"]

    export_fields = ["rep","dem","total","outs_rep","outs_dem","outs_total","proj_rep","proj_dem","proj_total"]

    for category in category_info:
        df = category_info[category]["df"]
        state_vals = df[export_fields].sum().to_dict()
        high_level_data.setdefault(election,{}).setdefault("state",{})[category] = state_vals
        for county, group in df.groupby(["county"]):
            county_vals = group[export_fields].sum().to_dict()
            high_level_data.setdefault(election,{}).setdefault("county",{}).setdefault(county,{})[category] = county_vals
        if category_info[category]["level"] == "precinct":
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
    category_info = load_virginia()

    #Cohn's guess is 77% - 23% for mail vote
    #and 54.5% - 45.5% for early vote
    #also 3.3 million votes total
    #-> election day = 3300000 - 310584 - 858571 = 2130845
    # election day 2020 was 1630415
    # -> election_day_ratio of 1.31
    #last piece is the election_day_margin_diff:
    #2130845 * x + 310584 * -0.54 + 858571 * -0.09 = 3300000 * 0.00 -> x = 0.115
    #in 2020 margins were:
    #absentee (not broken out) 947547 vs 1819198 -> -0.315
    #dayof: 1008098 vs 580837 -> 0.269


    settings = {
        "election_day_ratio": 1.31,
        "election_day_margin_diff": 0.115 - 0.269,
        "absentee_mail_ratio": 310584 / (1796973 + 1020821),
        "absentee_mail_diff": -0.54 + 0.315,
        "early_voting_ratio": 858571 / (1796973 + 1020821),
        "early_voting_diff": -0.09 + 0.315,
    }

    add_predictors(category_info)
    model_votes(category_info, settings)
    generate_predictions(category_info, settings)
    write_categories(category_info)
