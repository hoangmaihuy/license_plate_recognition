import os

xml_pred_dir = './data/plate_data/test/xml_pred'
output_dir = './data/plate_data/test/output'

def write_plate_data(xml_file, platetext, xmin, ymin, xmax, ymax):
    print("Writing to", xml_file)
    data = f'<annotation><object>' \
	       f'<platetext>{platetext}</platetext>' \
		   f'<bndbox>'\
		   f'<xmin>{xmin}</xmin>' \
		   f'<ymin>{ymin}</ymin>' \
		   f'<xmax>{xmax}</xmax>' \
		   f'<ymax>{ymax}</ymax>' \
		   f'</bndbox>' \
	       f'</object></annotation>'
    with open(xml_file, 'w') as f:
        f.write(data)

if __name__ == '__main__':
    # Read bouding box and platetext from .txt files in output folder
    # then write to xml file 
    if not os.path.exists(xml_pred_dir):
        os.makedirs(xml_pred_dir)
    for idx in range(1, 101):
        with open(os.path.join(output_dir, str(idx)+'.txt'), 'r') as f:
            xmin, ymin, xmax, ymax = map(int, f.readline().split())
        with open(os.path.join(output_dir, str(idx)+'_text.txt'), 'r') as f:
            label = f.readline()
        write_plate_data(os.path.join(xml_pred_dir, str(idx)+'.xml'), label, xmin, ymin, xmax, ymax)
            