import json 
import csv 

def get_dr_speech(received_data, dr_speaker_label):
    '''
        DESCRIPTION
        -------------
        Forms a list of the doctor's speech only. 
        Due to the nature of the IBM Watson STT transcription with speaker labels, it has to compare the time stamps between 
        the 'results' and 'speaker_labels' lists contained in the data in order to extract the word associated with the desired speaker label. 
        
        PARAMETERS
        -------------
            received_data: JSON  
                JSON file generated by the IBM Watson STT API, with speaker_labels=TRUE
            dr_speaker_label: int 
                Speaker label of doctor (identified from STT transcription manually)

        RETURNS
        -------------
            dr_only_list: list 
                The list contains the doctor's speech only 

    '''

    # Create a list of words and timestamps 
    words_and_stamps = []
    for result in received_data['results']:
        for time_stamps in result['alternatives'][0]['timestamps']:
            words_and_stamps.append(time_stamps)
    
    # Create a list of speaker_labels with the associated timestamps 
    speaker_label_list = []
    for speaker in received_data['speaker_labels']:
        speaker_label = speaker['speaker']
        from_label_timestamps = speaker['from']
        to_label_timestamps = speaker['to']
        speaker_label_list.append(f'Speaker {speaker_label} From {from_label_timestamps} To {to_label_timestamps}')
    
    # Create a list containing only the doctor's speech 
    dr_only_list = []
    for a in range(0, len(words_and_stamps)):
        if int(speaker_label_list[a].split(' ')[1]) == dr_speaker_label:
            word = words_and_stamps[a][0]
            s = speaker_label_list[a]
            dr_only_list.append(f'{word} {s}')

    return dr_only_list

# WRITE CSV FILE 
def write_csv(filename, dr_speech_list):
    '''
        DESCRIPTION
        -------------
        Writes a JSON file from IBM Watson STT API to a CSV file saved to the root directory, with headers: WORD, SPEAKER, FROM, TO. 
            Word = specific word spoken
            Speaker = speaker label
            From = timestamp for start time of word spoken 
            To = timestamp for end time of word spoken
        
        PARAMETERS
        -------------
            filename: str
                Specify name of new CSV file that will be created 
            dr_speech_list: list
                List containing only doctor's speech.

        RETURNS
        -------------
            None  
    '''
    # New CSV file name    
    new_csv_file = "" + filename + ".csv"         

    with open(new_csv_file, 'w', newline='') as f:   
        
        # Columm names for CSV data 
        fieldnames = ['WORD','SPEAKER', 'FROM', 'TO']       
        write_csv = csv.DictWriter(f, fieldnames=fieldnames)        
        write_csv.writeheader()

        # Write in data
        for i in range(0,len(dr_speech_list)):
            write_csv.writerow({'WORD': dr_speech_list[i].split(' ')[0], 'SPEAKER': dr_speech_list[i].split(' ')[2], 'FROM': dr_speech_list[i].split(' ')[4] , 'TO': dr_speech_list[i].split(' ')[6]})

    return None
