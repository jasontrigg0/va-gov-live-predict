import pandas as pd
import re

def diff_locality():
    df = pd.read_csv("results_2021.csv")
    df = df.rename(columns={"PoliticalParty": "party", "LocalityName": "county", "PrecinctName": "precinct"})
    #df["LocalityName"] = df["LocalityName"].str.upper()
    #df["PrecinctName"] = df["PrecinctName"].apply(lambda x: x.rsplit(" ",1)[0])
    l1 = set(list(df[["county","precinct"]].itertuples(index=False, name=None)))

    df = pd.read_csv("results_2020.csv")
    df = df.rename(columns={"PoliticalParty": "party"})
    df["precinct"] = df["precinct"].apply(map_precincts_2020_2021)
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



def merge_baselines(categories_live, categories_baseline):
    for category in categories_live:
        assert(categories_live[category]["level"] == categories_baseline[category]["level"])
        level = categories_live[category]["level"]
        if level == "precinct":
            join_cols = ["county","precinct"]
        elif level == "county":
            join_cols = ["county"]
        else:
            raise
        df_baseline = categories_baseline[category]["df"]

        df_baseline = df_baseline.rename(columns={x: "baseline_"+x for x in df_baseline.columns if x not in join_cols})

        df_live = categories_live[category]["df"]
        df_live = df_live.merge(df_baseline, how="left")

        #confirm all rows have baseline values
        invalid = df_baseline[pd.isna(df_baseline["baseline_margin"])]
        if len(invalid):
            print("Invalid values in baseline margin")
            print(invalid)
            raise

        #confirmed no NaN values in baseline above, so any NaN values must come from missing rows in the join
        unmatched = df_live[pd.isna(df_live["baseline_margin"])]
        if len(unmatched):
            print("Precincts from live dataset unmatched in the baseline dataset. Or possibly there's a math error somewhere causing NaNs in the join.")
            print(unmatched)
            raise

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

def load_election_live(filename, year, simulate=False):
    #load election info from the live feed
    df = pd.read_csv(filename)
    df = df.rename(columns={"PoliticalParty": "party", "LocalityName": "county", "PrecinctName": "precinct", "Votes": "votes"})
    df["state"] = "Virginia"
    df["county"] = df["county"].apply(lambda x: re.sub(" city$"," City",x).replace("King & Queen","King and Queen")) #rename counties to match the geojson file

    df = df[["party","state","county","precinct","votes"]]
    df = tally_votes(df)

    df["fit"] = 1 #leave out some precincts from the fits
    df["confirmed"] = 0 #allow confirmation that certain precinct data is correct

    #TODO: assert one row per (state, county, precinct) here

    #columns should be [state, county, precinct, rep, dem, other, fit, confirmed]
    assert(len(df.columns) == 8)
    if year == 2020:
        df = map_precincts_2020(df)
    elif year == 2021:
        df = map_precincts_2021(df)
    else:
        raise

    #VA records its absentee voting by (county,congressional district)
    #there are 95 counties and only 11 districts so this generally means one per county
    #but a few counties touch multiple districts -> multiple rows
    #however some counties are missing absentee rows entirely (why? is this an error?): add these if necessary

    #add missing absentee rows
    if year == 2020:
        norton_city_precincts = ["# AB - Central Absentee Precinct (09)","# EV - Central Absentee Precinct (09)","## Provisional"]
        for p in norton_city_precincts:
            df = df.append({"state": "Virginia", "county": "Norton City", "precinct": p, "dem": 0, "rep": 0, "other": 0}, ignore_index=True)
    elif year == 2021:
        pass
    else:
        raise


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
            include_clause = (any(fn(x["precinct"]) for fn in info["include"]) if info.get("include") else True)
            exclude_clause = (not any(fn(x["precinct"]) for fn in info.get("exclude",[])))
            return include_clause and exclude_clause
        output[category]["df"] = df[df.apply(filter_fn, axis=1)]
        if info["level"] == "precinct":
            pass
        elif info["level"] == "county":
            output[category]["df"].drop("precinct", axis=1)
        else:
            raise
    return output


def get_category_info(year):
    return {
        2020: {
            "Election Day Votes": {
                "level": "precinct",
                "exclude": [
                    lambda x: x.startswith("# AB - Central Absentee Precinct"),
                    lambda x: x.startswith("## Provisional")
                ]
            },
            "Provisional Votes": {
                "level": "precinct",
                "include": [lambda x: x.startswith("## Provisional")]
            },
            "Absentee by Mail Votes": {
                "level": "precinct",
                "include": [lambda x: x.startswith("# AB - Central Absentee Precinct")]
            },
            "Advanced Voting Votes": {
                "level": "precinct",
                "include": [lambda x: x.startswith("# EV - Central Absentee Precinct")]
            }
        },
        2021: {
            "Election Day Votes": {
                "level": "precinct",
                "exclude": [
                    lambda x: x.startswith("# AB - Central Absentee Precinct"),
                    lambda x: x.startswith("# EV - Central Absentee Precinct"),
                    lambda x: x.startswith("# PE - Central Absentee Precinct"),
                    lambda x: x.startswith("## Provisional")
                ]
            },
            "Provisional Votes": {
                "level": "precinct",
                "include": [lambda x: x.startswith("## Provisional")]
            },
            "Absentee by Mail Votes": {
                "level": "precinct",
                "include": [
                    lambda x: x.startswith("# AB - Central Absentee Precinct")
                ],
            },
            "Advanced Voting Votes": {
                "level": "precinct",
                "include": [lambda x: x.startswith("# EV - Central Absentee Precinct")]
            }
        }
    }[year]

def map_precincts_2020(df):
    #map from 2020 precincts to 2021
    rename_precincts = {
        #rename
        "201 - MAURY SCHOOL (08)": "201 - NAOMI L. BROOKS SCHOOL (08)",
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
        "203 - NI RIVER - ELYS FORD (07)": "203 - NI RIVER ELYS FORD (07)",
        #mergers
        "002 - NORTH (09)": "005 - WEST (09)",
        "003 - SOUTH (09)": "005 - WEST (09)",
        "301 - DENDRON (04)": "301 - DENDRON (04)",
        "302 - WALLS BRIDGE (04)": "301 - DENDRON (04)",
    }

    split_precincts = {
        "303 - RIDGE (09)": [
            {"name":"303 - RIDGE (09)", "frac": 0.5},
            {"name": "304 - LONGS FORK (09)", "frac": 0.5}
        ],
        "419 - LANE (08)": [
            {"name": "419 - LANE #1 (08)", "frac": 0.5},
            {"name": "430 - LANE #2 (08)", "frac": 0.5},
        ],
        "501 - BAILEYS (08)": [
            {"name": "501 - BAILEYS #1 (08)", "frac": 0.5},
            {"name": "531 - BAILEYS #2 (08)", "frac": 0.5}
        ],
        "516 - WEYANOKE (08)": [
            {"name": "516 - WEYANOKE #1 (08)", "frac": 0.5},
            {"name": "532 - WEYANOKE #2 (08)", "frac": 0.5},
        ],
        "522 - CAMELOT (11)": [
            {"name": "522 - CAMELOT #1 (11)", "frac": 0.5},
            {"name": "534 - CAMELOT #2 (11)", "frac": 0.5},
        ],
    }

    #rename + merge
    df["precinct"] = df.apply(lambda x: rename_precincts.get(x["precinct"], x["precinct"]), axis=1)
    df = df.groupby(["state","county","precinct"]).agg(sum).reset_index()

    #split
    def is_split(row):
        return row["precinct"] in split_precincts or row["precinct"].startswith("# AB - Central Absentee Precinct")
    def split(frame):
        assert(len(frame) == 1)
        row = frame.iloc[0]
        if row["precinct"] in split_precincts:
            output_rows = []
            for out in split_precincts[row["precinct"]]:
                split_row = row.copy()
                split_row["precinct"] = out["name"]
                split_row["rep"] *= out["frac"]
                split_row["dem"] *= out["frac"]
                split_row["other"] *= out["frac"]
                split_row["fit"] = 0 #don't fit these split precincts
                output_rows.append(split_row)
            return pd.DataFrame(output_rows)
        elif row["precinct"].startswith("# AB - Central Absentee Precinct"):
            output_rows = [
                row.copy(),
                row.copy()
            ]
            output_rows[1]["precinct"] = output_rows[1]["precinct"].replace("# AB - Central Absentee Precinct","# EV - Central Absentee Precinct")
            return pd.DataFrame(output_rows)
        else:
            return pd.DataFrame([row])

    #filter then split then combine because the split is kinda slow
    df_no_split = df[df.apply(lambda x: not is_split(x),axis=1)]
    df_split = df[df.apply(is_split,axis=1)]
    df_split = df_split.groupby(["state","county","precinct"]).apply(split).reset_index(drop=True)
    df = pd.concat([df_no_split, df_split])

    return df

def map_precincts_2021(df):
    #process 2021 precincts
    df["precinct"] = df["precinct"].apply(lambda x: x.replace("# PE - Central Absentee Precinct","# AB - Central Absentee Precinct"))
    df = df.groupby(["state","county","precinct"]).agg(sum).reset_index()
    return df

def tally_votes(df):
    cols = ["state","county","precinct"]
    rep_votes = df[df["party"] == "Republican"].groupby(cols)["votes"].sum().reset_index().rename(columns={"votes": "rep"})
    dem_votes = df[df["party"] == "Democratic"].groupby(cols)["votes"].sum().reset_index().rename(columns={"votes": "dem"})
    other_votes = df[(df["party"] != "Republican") & (df["party"] != "Democratic")].groupby(cols)["votes"].sum().reset_index().rename(columns={"votes": "other"})
    all_votes = rep_votes.merge(dem_votes,on=cols).merge(other_votes,on=cols)
    assert(len(all_votes) == max(len(rep_votes),len(dem_votes),len(other_votes)))

    return all_votes

def load_virginia():
    #diff_locality() #check on precinct changes between 2020 -> 2021
    categories_baseline = load_election_live("results_2020.csv",2020)
    compute_stats(categories_baseline)

    #categories_live = load_election_live("results_2021_8pm.csv",2021) #as of 8PM election night
    categories_live = load_election_live("results_2021.csv",2021) #latest
    compute_stats(categories_live)

    merge_baselines(categories_live, categories_baseline)

    return categories_live
