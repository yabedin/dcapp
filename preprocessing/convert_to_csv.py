import json 
import csv 

def convert_to_csv(received_data, filename):
    '''
    DESCRIPTION
    -------------
    Writes a JSON file from IBM Watson STT API to a CSV file saved to the root directory, with headers: WORD, SPEAKER, FROM, TO. 
    WORD = specific word spoken
    SPEAKER = speaker label
    FROM = timestamp for start time of word spoken 
    TO = timestamp for end time of word spoken

    PARAMETERS
    -------------
    received_data: dict
        Dict object sent to API via HTTP POST request
    filename: str
        Specify name of new CSV file that will be created 

    RETURNS
    -------------
    new_csv_file: csv
        CSV file created in root directory   
    '''
    
    # Check if data is in correct format 
    try:
        received_data['results']

        new_csv_file = filename + ".csv"

        with open(new_csv_file, 'w', newline='') as f:
            fieldnames = ['WORD', 'FROM', 'TO', 'SPEAKER']  # Columm names for CSV data 
            write_csv = csv.DictWriter(f, fieldnames=fieldnames)      

            write_csv.writeheader()
            
            for result in received_data['results']:
                for word in result['alternatives'][0]['timestamps']:
                    for i in range(len(received_data['speaker_labels'])):
                        if received_data['speaker_labels'][i]['from'] == word[1]:
                            write_csv.writerow({'WORD': word[0], 'FROM': word[1], 'TO': word[2], 'SPEAKER': received_data['speaker_labels'][i]['speaker']})
        
        f.close()  
        return new_csv_file

    # If any errors
    except:
        print("Exception occurred")
        return None
