import sys

# Get the file name from the command line arguments
file_name = sys.argv[1]
output_name = sys.argv[2]

# Open the input and output files
file = open(file_name, 'r')
outfile = open(output_name, 'w')

Opcodes = {
        'add': {'opcode': '0110011', 'funct7': '0000000', 'funct3': '000'},
        'sub': {'opcode': '0110011', 'funct7': '0100000', 'funct3': '000'},
        'sll': {'opcode': '0110011', 'funct7': '0000000', 'funct3': '001'},
        'slt': {'opcode': '0110011', 'funct7': '0000000', 'funct3': '010'},
        'sltu': {'opcode': '0110011', 'funct7': '0000000', 'funct3': '011'},
        'xor': {'opcode': '0110011', 'funct7': '0000000', 'funct3': '100'},
        'srl': {'opcode': '0110011', 'funct7': '0000000', 'funct3': '101'},
        'or': {'opcode': '0110011', 'funct7': '0000000', 'funct3': '110'},
        'and': {'opcode': '0110011', 'funct7': '0000000', 'funct3': '111'},
        'lw': {'opcode': '0000011', 'funct3': '010'},
        'addi': {'opcode': '0010011', 'funct3': '000'},
        'sltiu': {'opcode': '0010011', 'funct3': '011'},
        'jalr': {'opcode': '1100111', 'funct3': '000'},
        'sw': {'opcode': '0100011', 'funct3': '010'},
        'beq': {'opcode': '1100011', 'funct3': '000'},
        'bne': {'opcode': '1100011', 'funct3': '001'},
        'blt': {'opcode': '1100011', 'funct3': '100'},
        'bge': {'opcode': '1100011', 'funct3': '101'},
        'bltu': {'opcode': '1100011', 'funct3': '110'},
        'bgeu': {'opcode': '1100011', 'funct3': '111'},
        'lui': {'opcode': '0110111'},
        'auipc': {'opcode': '0010111'},
        'jal': {'opcode': '1101111'},
        'mul' : {'opcode': '1000000', 'funct7': '0000000', 'funct3': '000'},
        'rst' : {'opcode': '0000001', 'funct7': '0000000', 'funct3': '000'},
        'halt' : {'opcode': '0000000', 'funct7': '0000000', 'funct3': '000'},
        'rvrs' : {'opcode': '1111111', 'funct7': '0000000', 'funct3': '000'}
}

Registers = {
        'zero': ['x0', '00000'],
        'ra': ['x1', '00001'],
        'sp': ['x2', '00010'],
        'gp': ['x3', '00011'],
        'tp': ['x4', '00100'],
        't0': ['x5', '00101'],
        't1': ['x6', '00110'],
        't2': ['x7', '00111'],
        's0': ['x8', '01000'],
        'fp': ['x8', '01000'],
        's1': ['x9', '01001'],
        'a0': ['x10', '01010'],
        'a1': ['x11', '01011'],
        'a2': ['x12', '01100'],
        'a3': ['x13', '01101'],
        'a4': ['x14', '01110'],
        'a5': ['x15', '01111'],
        'a6': ['x16', '10000'],
        'a7': ['x17', '10001'],
        's2': ['x18', '10010'],
        's3': ['x19', '10011'],
        's4': ['x20', '10100'],
        's5': ['x21', '10101'],
        's6': ['x22', '10110'],
        's7': ['x23', '10111'],
        's8': ['x24', '11000'],
        's9': ['x25', '11001'],
        's10': ['x26', '11010'],
        's11': ['x27', '11011'],
        't3': ['x28', '11100'],
        't4': ['x29', '11101'],
        't5': ['x30', '11110'],
        't6': ['x31', '11111'],
        'x0': ['x0', '00000'],
        'x1': ['x1', '00001'],
        'x2': ['x2', '00010'],
        'x3': ['x3', '00011'],
        'x4': ['x4', '00100'],
        'x5': ['x5', '00101'],
        'x6': ['x6', '00110'],
        'x7': ['x7', '00111'],
        'x8': ['x8', '01000'],
        'x9': ['x9', '01001'],
        'x10': ['x10', '01010'],
        'x11': ['x11', '01011'],
        'x12': ['x12', '01100'],
        'x13': ['x13', '01101'],
        'x14': ['x14', '01110'],
        'x15': ['x15', '01111'],
        'x16': ['x16', '10000'],
        'x17': ['x17', '10001'],
        'x18': ['x18', '10010'],
        'x19': ['x19', '10011'],
        'x20': ['x20', '10100'],
        'x21': ['x21', '10101'],
        'x22': ['x22', '10110'],
        'x23': ['x23', '10111'],
        'x24': ['x24', '11000'],
        'x25': ['x25', '11001'],
        'sx26': ['x26', '11010'],
        'x27': ['x27', '11011'],
        'x28': ['x28', '11100'],
        'x29': ['x29', '11101'],
        'x30': ['x30', '11110'],
        'x31': ['x31', '11111'],
}

labels = {}
instructions = []
machine = []
lineNumber = 1
virtual_halt_found = 0
virtual_halt_line_number = 0
last_non_empty_line = 0

def extract_instructions():
    for line in file.readlines():
        line = line.strip()
        instructions.append(line)

def find_label():
    lineNumber = 1
    for i in range(len(instructions)):
        line = instructions[i].split()
        if(len(line) == 0):
            lineNumber += 1
            continue
        if(line[0][-1] == ':'): 
            # If label is present
            labels[line[0][:-1]] = lineNumber
            line.pop(0)
        line = ' '.join(line)
        instructions[i] = line
        lineNumber += 1

def error_gen1():
    # Syntax Error Type 1
    error_message = f"Line {lineNumber}: Invalid Syntax"
    print(error_message)
    sys.exit(0)

def error_gen2():
    # Invalid opcode
    error_message = f"Line {lineNumber}: Invalid Instruction"
    print(error_message)
    sys.exit(0)

def error_gen3(register):
    # Invalid register
    error_message = f"Line {lineNumber}: Invalid Register, {register}"
    print(error_message)
    sys.exit(0)

def error_gen4(imm):
    # invalid immediate
    error_message = f"Line {lineNumber}: Invalid Immediate, {imm}"
    print(error_message)
    sys.exit(0)

def error_gen5(imm):
    # Immediate outside the range
    error_message = f"Line {lineNumber}: Immediate outside the range, {imm}"
    print(error_message)
    sys.exit(0)

def error_gen6(label):
    # invalid label
    error_message = f"Line {lineNumber}: Invalid Label, {label}"
    print(error_message)
    sys.exit(0)    

def error_gen7():
    # check for virtual halt instruction
    if(virtual_halt_line_number != last_non_empty_line):
        error_message = f"Error: Virtual Halt Not Found As Last Instruction"
        print(error_message)
        sys.exit(0)

def to_bin(value, num_bits):
    if value >= 0:
        value = bin(value)[2:].zfill(num_bits)
    else:
        value = bin(value & (2**num_bits - 1))[2:].zfill(num_bits)
    return (value)

def check_reg(reg):
    # check whether register is valid or not
    if reg not in Registers:
        error_gen3(reg)
    return Registers[reg][1]

def check_imm(imm, num_bits, valid_bits):
    # check whether immediate is outside range or not
    if(type(imm) != type(0)):
        # imm is not an integer
        error_gen4(imm)
    if(imm < -(2**(valid_bits-1)) or imm >= 2**(valid_bits-1)):
        error_gen5(imm)
    return to_bin(imm, num_bits)

def check_label(label, num_bits, valid_bits):
    if(type(label) == type(0)):
        imm = check_imm(label, num_bits, valid_bits)
        # if label%4 != 0:
        #     error_gen6(label)
        return imm
    if label not in labels:
        error_gen6(label)
    labelLineNumber = labels[label]
    imm = (labelLineNumber-lineNumber)*4
    imm = check_imm(imm, num_bits, valid_bits)
    return imm

def check_virtual_halt(parts):
    global virtual_halt_found, virtual_halt_line_number
    opcode, rs1, rs2, label = parts
    condition1 = (rs1 == 'zero')
    condition2 = (rs2 == 'zero')
    condition3 = (label == 0)
    if(condition1 and condition2 and condition3):
        virtual_halt_found = 1
        virtual_halt_line_number = lineNumber

def partition(instruction):
    new_data = []
    elements = instruction.split(' ')
    ele2 = elements[1].split(',')
    if len(ele2) == 1:
        error_gen1()
    if elements[0] in ['lw', 'sw'] and len(ele2) != 2:
        error_gen1()
    elements = elements[:-1]
    if '(' in ele2[1]:
        ele3 = ele2[1].split('(')
        ele3[1] = ele3[1][:-1]
        ele2 = ele2[:-1]
        ele2.append(ele3[1])
        ele2.append(int(ele3[0]))
    else:
        for item in ele2:
            try:
                new_data.append(int(item))
            except ValueError:
                new_data.append(item)
        ele2 = new_data

    for i in ele2:
        elements.append(i)
    result_list = elements
    return result_list

def assemble_r_type(instruction):
    parts = partition(instruction)
    if(len(parts) != 4):
        error_gen1()
    opcode, rd, rs1, rs2 = parts
    rd = check_reg(rd)
    rs1 = check_reg(rs1)
    rs2 = check_reg(rs2)
    funct7 = Opcodes[opcode]['funct7']
    funct3 = Opcodes[opcode]['funct3']
    opcode_final = Opcodes[opcode]['opcode']
    machine_code = funct7+rs2+rs1+funct3+rd+opcode_final
    return machine_code

def assemble_i_type(instruction):
    parts = partition(instruction)
    if(len(parts) != 4):
        error_gen1()
    opcode, rd, rs1, imm = parts
    rd = check_reg(rd)
    rs1 = check_reg(rs1)
    imm = check_imm(imm, 12, 12) 
    funct3 = Opcodes[opcode]['funct3']
    opcode_final = Opcodes[opcode]['opcode']
    machine_code = imm+rs1+funct3+rd+opcode_final;
    return machine_code

def assemble_s_type(instruction):
    parts = partition(instruction)
    if(len(parts) != 4):
        error_gen1()
    opcode, rs2, rs1, imm = parts 
    rs1 = check_reg(rs1)
    rs2 = check_reg(rs2)
    imm = check_imm(imm, 12, 12)
    funct3 = Opcodes[opcode]['funct3']
    opcode_final = Opcodes[opcode]['opcode']
    imm_11_5 = imm[-12:-5]
    imm_4_0 = imm[-5:]
    machine_code = imm_11_5+rs2+rs1+funct3+imm_4_0+opcode_final;
    return machine_code

def assemble_b_type(instruction):
    parts = partition(instruction)
    check_virtual_halt(parts)
    if(len(parts) != 4):
        error_gen1()
    opcode, rs1, rs2, label = parts
    rs1 = check_reg(rs1)
    rs2 = check_reg(rs2)
    imm = check_label(label, 13, 12)
    funct3 = Opcodes[opcode]['funct3']
    opcode_final = Opcodes[opcode]['opcode']
    imm_12 = imm[-13]
    imm_11 = imm[-12]
    imm_10_5 = imm[-11:-5]
    imm_4_1 = imm[-5:-1]
    machine_code = imm_12+imm_10_5+rs2+rs1+funct3+imm_4_1+imm_11+opcode_final
    return machine_code

def assemble_u_type(instruction):
    parts = partition(instruction)
    if(len(parts) != 3):
        error_gen1()
    opcode, rd, imm = parts 
    rd = check_reg(rd)
    imm = check_imm(imm, 32, 20)
    opcode_final = Opcodes[opcode]['opcode']
    imm_31_12 = imm[-32:-12]
    machine_code = imm_31_12+rd+opcode_final;
    return machine_code

def assemble_j_type(instruction):
    parts = partition(instruction)
    if(len(parts) not in [3, 4]):
        error_gen1()
    if len(parts) == 3:
        opcode, rd, label = parts 
    if len(parts) == 4:
        opcode, rd, label = parts[0], parts[1], parts[3]
    rd = check_reg(rd)
    imm = check_label(label, 21, 20)
    opcode_final = Opcodes[opcode]['opcode']
    imm_20 = imm[-21]
    imm_10_1 = imm[-11:-1]
    imm_11 = imm[-12]
    imm_19_12 = imm[-20:-12]
    machine_code = imm_20+imm_10_1+imm_11+imm_19_12+rd+opcode_final;
    return machine_code

def assemble_mul(instruction):
    parts = partition(instruction)
    if(len(parts) != 4):
        error_gen1()
    opcode, rd, rs1, rs2 = parts
    rd = check_reg(rd)
    rs1 = check_reg(rs1)
    rs2 = check_reg(rs2)
    funct7 = Opcodes[opcode]['funct7']
    funct3 = Opcodes[opcode]['funct3']
    opcode_final = Opcodes[opcode]['opcode']
    machine_code = funct7+rs2+rs1+funct3+rd+opcode_final
    return machine_code

def assemble_rst(instruction):
    parts = partition(instruction)
    if(len(parts) != 1):
        error_gen1()
    opcode = parts[0]
    funct7 = Opcodes[opcode]['funct7']
    funct3 = Opcodes[opcode]['funct3']
    opcode_final = Opcodes[opcode]['opcode']
    machine_code = funct7+'00000'+'00000'+funct3+'00000'+opcode_final
    return machine_code

def assemble_halt(instruction):
    parts = partition(instruction)
    if(len(parts) != 1):
        error_gen1()
    opcode = parts[0]
    funct7 = Opcodes[opcode]['funct7']
    funct3 = Opcodes[opcode]['funct3']
    opcode_final = Opcodes[opcode]['opcode']
    machine_code = funct7+'00000'+'00000'+funct3+'00000'+opcode_final
    return machine_code

def assemble_rvrs(instruction):
    parts = partition(instruction)
    if(len(parts) != 3):
        error_gen1()
    opcode, rd, rs1 = parts
    rd = check_reg(rd)
    rs1 = check_reg(rs1)
    funct7 = Opcodes[opcode]['funct7']
    funct3 = Opcodes[opcode]['funct3']
    opcode_final = Opcodes[opcode]['opcode']
    machine_code = funct7+'00000'+rs1+funct3+rd+opcode_final
    return machine_code


def find_type(opcode):
    if opcode in ['add', 'sub', 'sll', 'slt', 'sltu', 'xor', 'srl', 'or', 'and']:
        return "R"
    elif opcode in ['lw', 'addi', 'sltiu', 'jalr']:
        return "I"
    elif opcode in ['sw']:
        return "S"
    elif opcode in ['beq', 'bne', 'blt', 'bge', 'bltu', 'bgeu']:
        return "B"
    elif opcode in ['lui', 'auipc']:
        return "U"
    elif opcode in ['jal']:
        return "J"
    elif opcode in ['mul']:
        return "MUL"
    elif opcode in ['rst']:
        return "RST"
    elif opcode in ['halt']:
        return "HALT"
    elif opcode in ['rvrs']:
        return "RVRS"
    else:
        error_gen2()

def main():
    global lineNumber, last_non_empty_line
    extract_instructions()
    find_label()

    for instruction in instructions:
        line = instruction.split()
        if len(line) == 0:
            # empty line
            lineNumber += 1
            continue
        if len(line) == 2:
            last_non_empty_line = lineNumber
            instruction_type = find_type(line[0])
            if(instruction_type == "R"):
                machine_code = assemble_r_type(instruction)
            elif(instruction_type == "I"):
                machine_code = assemble_i_type(instruction)
            elif(instruction_type == "S"):
                machine_code = assemble_s_type(instruction)
            elif(instruction_type == "B"):
                machine_code = assemble_b_type(instruction)
            elif(instruction_type == "U"):
                machine_code = assemble_u_type(instruction)
            elif(instruction_type == "J"):
                machine_code = assemble_j_type(instruction)
            elif(instruction_type == "MUL"):
                machine_code = assemble_mul(instruction)   
            elif(instruction_type == "RST"):
                machine_code = assemble_rst(instruction)
            elif(instruction_type == "HALT"):
                machine_code = assemble_halt(instruction)
            elif(instruction_type == "RVRS"):
                machine_code = assemble_rvrs(instruction)
            machine.append(machine_code)
        else:
            error_gen1()
        lineNumber += 1

    error_gen7()

    for i, line in enumerate(machine):
        # Check if it's the last line
        if i == len(machine) - 1:
            outfile.write(line)
        else:
            outfile.write(line + '\n')


if __name__ == "__main__":
    main()
    file.close()
    outfile.close()
    