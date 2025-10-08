import csv

def read_csv(file_path):
    row_list = []
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            row_list.append(row)
    return row_list

def update_mean_by_sample(nir_rows):
    nir_rows_with_mean_values = []
    for i in range(1, len(nir_rows)):
        if i % 2 == 0: continue
        nir_sample_mean_value = []
        for j in range(1, len(nir_rows[i])):
            mean_value = (float(nir_rows[i][j].replace(",", ".")) + float(nir_rows[i + 1][j].replace(",", "."))) / 2
            nir_sample_mean_value.append(mean_value)
        nir_rows_with_mean_values.append(nir_sample_mean_value)
    return nir_rows_with_mean_values

def write_to_csv(element, sample_size, rows, headers):
    headers.append(element)
    with open(f"./generate_datasets/{element.split('(')[0]}_{sample_size}_samples", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

def generate_datasets(result_rows, nir_rows, headers):
    #iterate over the headers of results, ignoring the first and the last two
    for i in range(1, len(result_rows[0]) - 2):
        dataset_lines = []
        # iterating over the lines of result_rows
        for j in range(1, len(result_rows)):
            #if no value is provided we wont add to the dataset
            if result_rows[j][i] != '0,00':
                value = float(result_rows[j][i].replace(",", "."))
                row = [result_rows[j][0]] + nir_rows[j - 1] + [value]
                dataset_lines.append(row)
        write_to_csv(result_rows[0][i], len(dataset_lines), dataset_lines, headers)

def handler():
    result_rows = read_csv('data_results.csv')
    nir_rows = read_csv('data_nir_187_samples.csv')
    nir_rows_means = update_mean_by_sample(nir_rows)
    generate_datasets(result_rows, nir_rows_means, nir_rows[0])
    print(result_rows)


if __name__ == '__main__':
    handler()