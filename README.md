Data sources:
geojson from: https://github.com/kjhealy/us-county/blob/master/data/geojson/gz_2010_us_050_00_20m.json
Virginia fips=51, eg see here: https://www.mcc.co.mercer.pa.us/dps/state_fips_code_listing.htm
python -c 'import json; counties = json.loads(open("gz_2010_us_050_00_20m.json",encoding="ISO-8859-1").read()); counties["features"] = [x for x in counties["features"] if x["properties"]["STATE"] == "51"]; print(json.dumps(counties))' > virginia-counties.json# va-gov-live-predict
