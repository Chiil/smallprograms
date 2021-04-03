import test_dict

input_dict = { 'itot' : 4, 'ktot' : 2}
output_dict = test_dict.read_dict(input_dict)

field = output_dict['field']
print(field)

field[:, :] = 2.*field[:, :]
print(field)

test_dict.print_data()

