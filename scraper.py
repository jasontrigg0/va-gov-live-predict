from bs4 import BeautifulSoup
import requests
import re
import csv
import pandas as pd
import json

def scrape_live_json(election_name, office):
    #https://results.elections.virginia.gov/vaelections/2020%20November%20General/Json/Locality/ACCOMACK_COUNTY/President_and_Vice_President.json
    fips = pd.read_csv("va_fips.csv")

    df = pd.DataFrame()

    for _,r in fips.iterrows():
        mappings = {"King and Queen County": "King & Queen County"}
        region = mappings.get(r["region"],r["region"])
        race = office
        url = f"https://results.elections.virginia.gov/vaelections/{election_name}/Json/Locality/{region.upper().replace(' ','_')}/{race.replace(' ','_')}.json"
        print(url)

        dat = json.loads(requests.get(url).text)
        for p in dat["Precincts"]:
            for c in p["Candidates"]:
                df = df.append({"LocalityName": region, "PrecinctName": p["PrecinctName"], **c}, ignore_index=True)
        print(df)
    df.to_csv("results_from_json.csv",index=False)

def download_live_misc_csv():
    #https://results.elections.virginia.gov/vaelections/2021%20June%20Special%20Election/Site/Statistics/Index.html
    #these have miscellaneous data, how to use?
    pass

def scrape_early_voting_stats():
    #can't find this information outside of vpap.org
    #maybe they're using the voter file, which isn't public information
    #https://www.vpap.org/elections/early-voting/november-2021-election/
    #https://www.vpap.org/voterfile/

    url = "https://www.vpap.org/elections/early-voting/november-2021-election/"
    r = requests.get(url)
    total_inperson, total_mail = re.findall("value:.*?(\d+)",r.text)[:2]
    print("totals:",total_inperson,total_mail)

    df = pd.read_csv("va_fips.csv")

    def get_counts(x):
        print(x["region"])
        mappings = {"King and Queen County": "King Queen County"}
        name = mappings.get(x["region"],x["region"])
        url = f"https://www.vpap.org/elections/early-voting/november-2021-election/locality-{name.lower().replace(' ','-')}-va/"
        r = requests.get(url)
        inperson, mail = re.findall("value:.*?(\d+)",r.text)[:2]
        x["inperson"] = int(inperson)
        x["mail"] = int(mail)
        return x

    df = df.apply(get_counts, axis=1)
    df.to_csv("early_voting.csv",index=False)

    assert(total_inperson == df["inperson"].sum())
    assert(total_mail == df["mail"].sum())

if __name__ == "__main__":
    #scrape_early_voting_stats()
    #scrape_live_json("2020 November General", "President and Vice President")
    scrape_live_json("2021 November General", "Governor")
