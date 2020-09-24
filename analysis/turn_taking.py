import csv
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import io
import base64

# Set backend to non-interactive
matplotlib.use('Agg')

class TurnTakingIndividual:

    def __init__(self, filepath, speaker_x=0, speaker_y=1):
        self.filepath = filepath
        self.speaker_x = speaker_x
        self.speaker_y = speaker_y

        self.speaker_list = []
        self.all_list = []
    
    def open_csv(self):
        with open(self.filepath, newline='') as csvfile:
            data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in data:
                joined_row = ' , '.join(row)
                self.speaker_list.append(joined_row)
        return self.speaker_list

    def convert_to_list(self):
        for row in self.open_csv():
            individual_items = row.split(',')
            if individual_items[0].split(',')[0] == 'WORD':
                pass
            else:
                self.all_list.append(individual_items)
    
        return self.all_list

    def get_individual_turn_lengths(self):
        doc_turn = 0 
        self.doc_turn_lengths = {}

        patient_turn = 0
        self.patient_turn_lengths = {}

        doc_turn_sum = 0
        patient_turn_sum = 0 

        for item in self.convert_to_list():
            if int(item[3]) == 0:
                doc_turn_sum = 0
                
                if patient_turn_sum == 0:
                    patient_turn += 1
                else:
                    pass

                patient_turn_sum += (float(item[2]) - float(item[1]))
                self.patient_turn_lengths[patient_turn] = patient_turn_sum
            
            else:
                patient_turn_sum = 0
                if doc_turn_sum == 0:
                    doc_turn += 1
                else:
                    pass
                doc_turn_sum += (float(item[2]) - float(item[1]))
                self.doc_turn_lengths[doc_turn] = doc_turn_sum
            
        # Plot results 
        plt.style.use('seaborn-deep')

        plt.bar(range(len(self.patient_turn_lengths)), list(self.patient_turn_lengths.values()), label='Doctor', width=0.8)
        plt.bar(range(len(self.doc_turn_lengths)), list(self.doc_turn_lengths.values()), label='Patient', width=0.4)

        plt.xticks(range(len(self.doc_turn_lengths)), list(self.doc_turn_lengths.keys()))

        plt.xlabel("Turn number")
        plt.ylabel('Length of time spoken (seconds)')
        plt.title('Spoken time (s) per turn for each speaker')
        plt.legend(loc='best')

        # plt.show()

        # Encode image as base64
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight')
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode()

        return plot_url



        