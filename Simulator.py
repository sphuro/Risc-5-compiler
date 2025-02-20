import sys

Memory = {}
mem_add = 2**16 # mem_add : 0x001_0000, this number from hex to decimal is 2**16
for i in range(32): # as there are 32 memory locations
    Memory[mem_add] = 0 # initialising every mem address with value 0
    mem_add += 4 # moving to next mem address

Registers = { 
    # dictionary to track values of registers
    # initialising with all zeros
    '00000' : 0, 
    '00001' : 0,
    '00010' : 0,
    '00011' : 0,
    '00100' : 0,
    '00101' : 0,
    '00110' : 0,
    '00111' : 0,
    '01000' : 0,
    '01001' : 0,
    '01010' : 0,
    '01011' : 0,
    '01100' : 0,
    '01101' : 0,
    '01110' : 0,
    '01111' : 0,
    '10000' : 0,
    '10001' : 0,
    '10010' : 0,
    '10011' : 0,
    '10100' : 0,
    '10101' : 0,
    '10110' : 0,
    '10111' : 0,
    '11000' : 0,
    '11001' : 0,
    '11010' : 0,
    '11011' : 0,
    '11100' : 0,
    '11101' : 0,
    '11110' : 0,
    '11111' : 0
}

PC = 0 # program counter

instructions = []
output = []
virtual_hault_found = False

# Get the file name from the command line arguments
file_name = sys.argv[1]
output_name = sys.argv[2]

# Open the input and output files
file = open(file_name, 'r')
outfile = open(output_name, 'w')

def extract_instructions():
    for line in file.readlines():
        line = line.strip()
        instructions.append(line)

def get_output1():
    out_str = "0b"+to_bin(PC, 32)
    for i in Registers.values():
        out_str += " 0b"+to_bin(i, 32)
    out_str += " "
    output.append(out_str)

def get_output2():
    for i in Memory:
        out_str = "0x"+to_hex(i, 8)
        out_str += ":0b"+to_bin(Memory[i], 32)
        output.append(out_str)

def keep_zero():
    Registers['00000'] = 0

def init_sp():
    Registers['00010'] = 2**8
    
def ones_complement(binary):
    n = len(binary)
    binary = list(binary)
    for i in range(n):
        if binary[i] == '0':
            binary[i] = '1' 
        else:
            binary[i] = '0'
    binary = ''.join(binary)
    return binary
  
def to_decimal(binary):
    # Converts given binary string in 2's complement representation to decimal value
    n = len(binary)
    is_neg = False
    if binary[0] == '1':
        is_neg = True
        binary = ones_complement(binary)
    decimal_val = 0
    for i in range(n):
        idx = -(i+1)
        if binary[idx] == '1':
            decimal_val += 2**i 
    if is_neg:
        decimal_val += 1 
        decimal_val = -decimal_val
    return decimal_val


def to_bin(value, num_bits):
    # converts decimal value to 2's complent binary representation
    if value >= 0:
        value = bin(value)[2:].zfill(num_bits)
    else:
        value = bin(value & (2**num_bits - 1))[2:].zfill(num_bits)
    return (value)

# convert decimal representation of signed binary to decimal representation of unsigned binary
def unsigned(val):
    val = to_bin(val, 32)
    unsigned_val = 0
    for i in range(32):
        idx = -(i+1)
        if val[idx] == '1':
            unsigned_val += 2**i 
    return unsigned_val

def to_hex(value, size):
    tmp_hex = hex(value)
    tmp_hex = tmp_hex[2:]
    to_add = size-len(tmp_hex)
    final_hex = "0"*to_add + tmp_hex
    return final_hex

# R Type Instructions
class R_type:
    def __init__(self, binary_instruction):
        self.funct7 = binary_instruction[:-25]
        self.rs2 = binary_instruction[-25:-20]
        self.rs1 = binary_instruction[-20:-15]
        self.funct3 = binary_instruction[-15:-12]
        self.rd = binary_instruction[-12:-7]
        self.opcode = binary_instruction[-7:]

    def add(self): 
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        Registers[self.rd] = rs1_value + rs2_value
    
    def sub(self):
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        Registers[self.rd] = rs1_value - rs2_value

    def sll(self):
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        Registers[self.rd] = rs1_value << unsigned(rs2_value & 0x1F)

    def slt(self):
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        if rs1_value < rs2_value:
            Registers[self.rd] = 1

    def sltu(self): 
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        if unsigned(rs1_value) < unsigned(rs2_value):
            Registers[self.rd] = 1

    def xor(self):
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        Registers[self.rd] = rs1_value ^ rs2_value

    def srl(self): 
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        Registers[self.rd] = rs1_value >> unsigned(rs2_value & 0x1F)

    def or_(self):
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        Registers[self.rd] = rs1_value | rs2_value

    def and_(self):
        rs1_value = Registers[self.rs1]
        rs2_value = Registers[self.rs2]
        Registers[self.rd] = rs1_value & rs2_value

    def simulate(self):
        global PC
        if self.funct3 == '000':
            if self.funct7 == '0000000':
                self.add()
            elif self.funct7 == '0100000':
                self.sub()
        elif self.funct3 == '001':
            self.sll()
        elif self.funct3 == '010':
            self.slt()
        elif self.funct3 == '011':
            self.sltu()
        elif self.funct3 == '100':
            self.xor()
        elif self.funct3 == '101':
            self.srl()
        elif self.funct3 == '110':
            self.or_()
        elif self.funct3 == '111':
            self.and_()
        PC = PC + 4

# I type Instructions
class I_type:
    def __init__(self, binary_instruction):
        self.imm = binary_instruction[:-20]
        self.imm = to_decimal(self.imm)
        self.rs1 = binary_instruction[-20:-15]
        self.funct3 = binary_instruction[-15:-12]
        self.rd = binary_instruction[-12:-7]
        self.opcode = binary_instruction[-7:]

    def addi(self):
        global PC
        rs1_value = Registers[self.rs1]
        Registers[self.rd] = rs1_value + self.imm
        PC = PC + 4

    def sltiu(self):
        global PC
        rs1_value = Registers[self.rs1]
        if unsigned(rs1_value) < unsigned(self.imm):
            Registers[self.rd] = 1
        PC = PC + 4

    def load(self):
        global PC
        mem_add = Registers[self.rs1] + self.imm
        Registers[self.rd] = Memory[mem_add]
        PC = PC + 4

    def jalr(self):
        global PC
        Registers[self.rd] = PC + 4
        PC = Registers[self.rs1] + self.imm
        PC = PC & 0xFFFFFFFE

    def simulate(self):
        if self.funct3 == '000' and self.opcode == '0010011':
            self.addi()
        elif self.funct3 == '011':
            self.sltiu()
        elif self.funct3 == '010':
            self.load()
        elif self.funct3 == '000' and self.opcode == '1100111':
            self.jalr()

# S type Instructions
class S_type:
    def __init__(self, binary_instruction):
        self.imm = binary_instruction[:-25] + binary_instruction[-12:-7]
        self.imm = to_decimal(self.imm)
        self.rs2 = binary_instruction[-25:-20]
        self.rs1 = binary_instruction[-20:-15]
        self.funct3 = binary_instruction[-15:-12]
        self.opcode = binary_instruction[-7:0]

    def store(self):
        mem_add = Registers[self.rs1] + self.imm
        Memory[mem_add] = Registers[self.rs2]

    def simulate(self):
        global PC
        self.store()
        PC = PC + 4

# B Type Instructions
class B_type:
    def __init__(self, binary_instruction):
        global virtual_hault_found
        self.opcode = binary_instruction[-7:]
        self.funct3 = binary_instruction[-15:-12]
        self.rs1 = binary_instruction[-20:-15]
        self.rs2 = binary_instruction[-25:-20]
        imm_11 = binary_instruction[-8]
        imm_4_1 = binary_instruction[-12:-8]
        imm_10_5 = binary_instruction[-31:-25]
        imm_12 = binary_instruction[-32]
        self.imm = imm_12+imm_11+imm_10_5+imm_4_1+"0"
        self.imm = to_decimal(self.imm)

        if binary_instruction == '00000000000000000000000001100011':
            virtual_hault_found = True

    def beq(self):
        global PC
        if Registers[self.rs1] == Registers[self.rs2]:
            PC = PC + self.imm
        else:
            PC = PC + 4
    
    def bne(self):
        global PC
        if Registers[self.rs1] != Registers[self.rs2]:
            PC = PC + self.imm
        else:
            PC = PC + 4
            
    def bge(self):
        global PC
        if Registers[self.rs1] >= Registers[self.rs2]:
            PC = PC + self.imm
        else:
            PC = PC + 4

    def bgeu(self):
        global PC
        if unsigned(Registers[self.rs1]) >= unsigned(Registers[self.rs2]):
            PC = PC + self.imm
        else:
            PC = PC + 4

    def blt(self):
        global PC
        if Registers[self.rs1] < Registers[self.rs2]:
            PC = PC + self.imm
        else:
            PC = PC + 4

    def bltu(self):
        global PC
        if unsigned(Registers[self.rs1]) < unsigned(Registers[self.rs2]):
            PC = PC + self.imm
        else:
            PC = PC + 4

    def simulate(self):
        if self.funct3 == '000':
            self.beq()
        elif self.funct3 == '001':
            self.bne()
        elif self.funct3 == '100':
            self.blt()
        elif self.funct3 == '101':
            self.bge()
        elif self.funct3 == '110':
            self.bltu()
        elif self.funct3 == '111':
            self.bgeu()  
        
# U Type instructions
class U_type:
    def __init__(self, binary_instruction):
        self.imm = binary_instruction[:-12] + '0'*12
        self.imm = to_decimal(self.imm)
        self.rd = binary_instruction[-12:-7]
        self.opcode = binary_instruction[-7:]

    def aupic(self):
        Registers[self.rd] = PC + self.imm
    
    def lui(self):
        Registers[self.rd] = self.imm
    
    def simulate(self):
        global PC
        if self.opcode == '0010111':
            self.aupic()
        elif self.opcode == '0110111':
            self.lui()
        PC = PC + 4


#  J type Instructions
class J_type:
    def __init__(self, binary_instruction):
        imm_20 = binary_instruction[-32]
        imm_10_1 = binary_instruction[-31:-21]
        imm_11 = binary_instruction[-21]
        imm_19_12 = binary_instruction[-20:-12]
        self.imm = imm_20+imm_19_12+imm_11+imm_10_1+'0'
        self.imm = to_decimal(self.imm)
        self.rd = binary_instruction[-12:-7]
        self.opcode = binary_instruction[-7:]

    def jal(self):
        global PC
        Registers[self.rd] = PC + 4
        PC = PC + self.imm 
        PC = PC & 0xFFFFFFFE
    
    def simulate(self):
        self.jal()

# Bonus Instructions
class Bonus:
    def __init__(self, binary_instruction):
        self.funct7 = binary_instruction[:-25]
        self.rs2 = binary_instruction[-25:-20] # For instructions with rs2
        self.rs1 = binary_instruction[-20:-15] # For instructions with rs1
        self.funct3 = binary_instruction[-15:-12]
        self.rd = binary_instruction[-12:-7] # For instructions with rd
        self.opcode = binary_instruction[-7:]

    def mul(self):
        global PC 
        Registers[self.rd] = Registers[self.rs1]*Registers[self.rs2]
        PC = PC + 4

    def rst(self):
        global PC
        for i in Registers:
            if i == '00010':
                Registers[i] = 2**8
            else:
                Registers[i] = 0
        PC = PC + 4

    def halt(self):
        global virtual_hault_found
        virtual_hault_found = True 
    
    def rvrs(self):
        global PC
        rs_value = Registers[self.rs1]
        rs_value = to_bin(rs_value)
        rs_value = rs_value[::-1]
        rs_value = to_decimal(rs_value)
        Registers[self.rd] = rs_value
        PC = PC + 4

    def simulate(self):
        if self.opcode == '1000000':
            self.mul()
        elif self.opcode == '0000001':
            self.rst()
        elif self.opcode == '0000000':
            self.halt()
        elif self.opcode == '1111111':
            self.rvrs()

def simulate(binary_instruction):
    global PC
    opcode = binary_instruction[-7:]
    if opcode == '0110011':
        r_inst = R_type(binary_instruction)
        r_inst.simulate()
    elif opcode == '0010011' or opcode == '0000011' or opcode == '1100111':
        i_inst = I_type(binary_instruction)
        i_inst.simulate()
    elif opcode == '0100011':
        s_inst = S_type(binary_instruction)
        s_inst.simulate()
    elif opcode == '0110111' or opcode == '0010111':
        u_inst = U_type(binary_instruction)
        u_inst.simulate()
    elif opcode == '1101111':
        j_inst = J_type(binary_instruction)
        j_inst.simulate()
    elif opcode == '1100011':
        b_inst = B_type(binary_instruction)
        b_inst.simulate()
    elif opcode in ['1000000', '0000001', '0000000', '1111111']:
        bonus_inst = Bonus(binary_instruction)
        bonus_inst.simulate()
    keep_zero()
    get_output1()

def main():
    global virtual_hault_found
    extract_instructions() # Creates list of input binary instructions
    init_sp()
    while(virtual_hault_found == False):
        instructions_idx = PC//4
        binary_instruction = instructions[instructions_idx]
        simulate(binary_instruction)
    get_output2()
    for i, line in enumerate(output):
        outfile.write(line + '\n')

if __name__ == "__main__":
    main()
    file.close()
    outfile.close()