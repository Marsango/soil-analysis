import csv

def read_csv(file_path):
    row_list = []
    with open(file_path, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=',')
        for row in csv_reader:
            row_list.append(row)
    return row_list

def update_mean_by_sample_result(result_rows):
    nir_rows_with_mean_values = []
    for i in range(1, len(result_rows)):
        if i % 2 == 0: continue
        nir_sample_mean_value = [result_rows[i][0]]
        for j in range(1, len(result_rows[i])):
            mean_value = (float(result_rows[i][j].replace(",", ".")) + float(result_rows[i + 1][j].replace(",", "."))) / 2
            nir_sample_mean_value.append(mean_value)
        nir_rows_with_mean_values.append(nir_sample_mean_value)
    return nir_rows_with_mean_values

def update_mean_by_sample_nir(nir_rows):
    nir_rows_with_mean_values = {}
    for i in range(1, len(nir_rows)):
        if i % 2 == 0: continue
        nir_sample_mean_value = []
        for j in range(1, len(nir_rows[i])):
            mean_value = (float(nir_rows[i][j].replace(",", ".")) + float(nir_rows[i + 1][j].replace(",", "."))) / 2
            nir_sample_mean_value.append(mean_value)
        sample_prefix = "Amostra " if "amostra" in nir_rows[i][0].lower() else ""
        sample_id = sample_prefix + "".join(c for c in nir_rows[i][0] if c.isdigit())
        nir_rows_with_mean_values[sample_id] = nir_sample_mean_value
    return nir_rows_with_mean_values

def write_to_csv(element, sample_size, rows, headers):
    new_headers = headers.copy()
    new_headers.append("result")
    with open(f"generated_datasets/{element.split('(')[0].strip()}_{sample_size}_samples.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(new_headers)
        writer.writerows(rows)

def generate_datasets(result_rows, nir_rows, headers):
    #iterate over the headers of results, ignoring the first (sample name)
    for i in range(1, len(result_rows[0])):
        dataset_lines = []
        # iterating over the lines of result_rows
        for j in range(1, len(result_rows)):
            #if no value is provided we won't add to the dataset
            if result_rows[j][i] != '0,00':
                value = result_rows[j][i]
                if type(value) == str:
                    value = float(value.replace(",", "."))
                row = []
                row.append(result_rows[j][0])
                sample_prefix = "Amostra " if "amostra" in result_rows[j][0].lower() else ""
                sample_id = sample_prefix + "".join(c for c in result_rows[j][0] if c.isdigit())
                row.extend(nir_rows[sample_id])
                row.append(value)
                dataset_lines.append(row)
        write_to_csv(result_rows[0][i], len(dataset_lines), dataset_lines, headers)

def handler():
    result_rows = read_csv('data_results.csv')
    result_rows.extend(update_mean_by_sample_result(read_csv('data_results_2.csv'))[1:])
    nir_rows = read_csv('data_nir_187_samples.csv')
    nir_rows.extend(read_csv("data_nir_67_samples.csv")[1:])
    nir_rows_means = update_mean_by_sample_nir(nir_rows)
    generate_datasets(result_rows, nir_rows_means, nir_rows[0])

if __name__ == '__main__':
    handler()