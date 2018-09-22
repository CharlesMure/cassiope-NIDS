import time
import os


def main():

    # Set the filename and open the file
    filename = 'collect.log'
    file = open(filename, 'r')

    ct_dst_sport_ltm_list = []
    ct_dst_src_ltm_list = []
    ct_src_dport_ltm_list = []
    ct_dst_ltm_list = []

    ct_dst_sport_ltm_hash = 0
    ct_dst_src_ltm_hash = 0
    ct_src_dport_ltm_hash = 0
    ct_dst_ltm_hash = 0
    ct_state_ttl = 0

    for i in range(100):
        ct_dst_sport_ltm_list.append(0)
        ct_dst_src_ltm_list.append(0)
        ct_src_dport_ltm_list.append(0)
        ct_dst_ltm_list.append(0)

    while 1:
        where = file.tell()
        line = file.readline()
        if not line:
            time.sleep(1)
            file.seek(where)
        else:
            logline = line[:-2]
            logline = logline.split(",")
            logline = removeNull(logline)

            #print(logline)

            ct_dst_sport_ltm_hash = hash(logline[1] + logline[2])
            ct_dst_sport_ltm_list.pop()
            ct_dst_sport_ltm_list.insert(0, ct_dst_sport_ltm_hash)

            ct_dst_src_ltm_hash = hash(logline[0] + logline[1])
            ct_dst_src_ltm_list.pop()
            ct_dst_src_ltm_list.insert(0, ct_dst_src_ltm_hash)

            ct_src_dport_ltm_hash = hash(logline[0] + logline[3])
            ct_src_dport_ltm_list.pop()
            ct_src_dport_ltm_list.insert(0, ct_src_dport_ltm_hash)

            ct_dst_ltm_hash = hash(logline[1])
            ct_dst_ltm_list.pop()
            ct_dst_ltm_list.insert(0, ct_dst_ltm_hash)

            ct_state_ttl = 0
            if (logline[4] == 62 or logline[4] == 63 or logline[4] == 254 or logline[4] == 255) and (logline[5] == 252 or logline[5] == 253) and logline[6] == "FIN":
                ct_state_ttl = 1

            elif (logline[4] == 0 or logline[4] == 62 or logline[4] == 254) and (logline[5] == 0) and logline[6] == "INT":
                ct_state_ttl = 2

            elif(logline[4] == 62 or logline[4] == 254) and (logline[5] == 60 or logline[5] == 252 or logline[5] == 253) and logline[6] == "CON":
                ct_state_ttl = 3
                
            elif(logline[4] == 254) and (logline[5] == 252) and logline[6] == "ACC":
                ct_state_ttl = 4
                
            elif(logline[4] == 254) and (logline[5] == 252) and logline[6] == "CLO":
                ct_state_ttl = 5
                
            elif(logline[4] == 254) and (logline[5] == 0) and logline[6] == "REQ":
                ct_state_ttl = 7
                
            else:
                ct_state_ttl = 0

            for i in range(7):
                logline.pop(0)

            dataline = logline + [str(ct_dst_ltm_list.count(ct_dst_ltm_hash))] + [str(ct_src_dport_ltm_list.count(ct_src_dport_ltm_hash))] + [
                str(ct_dst_sport_ltm_list.count(ct_dst_sport_ltm_hash))] + [str(ct_dst_src_ltm_list.count(ct_dst_src_ltm_hash))] + [str(ct_state_ttl)]

            #print("new data: ")
            #print(" ".join(dataline))  # already has newline

            write_data(dataline)


def write_data(line_data):
    with open("dataset.log", mode='a') as file:
        file.write(",".join(line_data) + "\n")

def removeNull(row):
    return ['0' if i=='' else i for i in row]


if __name__ == "__main__":
    main()
