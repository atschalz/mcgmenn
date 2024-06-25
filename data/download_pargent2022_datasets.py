import xmltodict
import os
import arff # make sure to pip install liac-arff
import urllib.request
import urllib3
import pandas as pd

def getxml(url):
    # From https://stackoverflow.com/questions/24124643/parse-xml-from-url-into-python-object
    http = urllib3.PoolManager()

    response = http.request('GET', url)
    try:
        data = xmltodict.parse(response.data)
    except:
        print("Failed to parse xml from response (%s)" % traceback.format_exc())
    return data

dataset_ids = {"Click_prediction_small": 41434,
               "Amazon_employee_access": 4135,
               "video-game-sales": 41216,
               "hpc-job-scheduling": 41212,
               "road-safety-drivers-sex": 41447,
               "open_payments": 41442,
               "okcupid-stem": 41440,
               "Midwest_survey": 41446,
               "Diabetes130US": 4541,
               "KDDCup09_upselling": 1114,
               "adult": 1590,
               "kdd_internet_usage": 981,
               "churn": 41283,
               "porto-seguro": 41224,
               "kick": 41162,
               "eucalyptus": 188,
               "wine-reviews": 41437,
               "medical_charges": 41444,
               "avocado-sales": 41210,
               "employee_salaries": 41445,
               "particulate-matter-ukair-2017": 41267,
               "flight-delay-usa-dec-2017": 41251,
               "nyc-taxi-green-dec-2016": 41255,
               "ames-housing": 41211
}

if __name__ == '__main__':
    for dataset_name in dataset_ids:
        if not os.path.exists(f"./raw/{dataset_name}"):
            os.mkdir(f"./raw/{dataset_name}")

        if not os.path.exists(f"./raw/{dataset_name}/{dataset_name}.csv"):
            print(f"Download {dataset_name} dataset...")
            xml_data = getxml(f"https://api.openml.org/api/v1/data/{dataset_ids[dataset_name]}")
            url = xml_data['oml:data_set_description']["oml:url"]

            urllib.request.urlretrieve(url, f"./raw/{dataset_name}/{dataset_name}.arff")

            try:
                dataset = arff.load(open(f"./raw/{dataset_name}/{dataset_name}.arff", 'rt'))
            except:
                print(f"Unable to download {dataset_name}")
                continue
            data = pd.DataFrame(dataset['data'], columns=[i[0] for i in dataset["attributes"]])

            data.to_csv(f"./raw/{dataset_name}/{dataset_name}.csv")
            print(f"Finished {dataset_name} dataset Download")

        else:
            print(f"Dataset {dataset_name} alreay exists.")





