import os
import re
import argparse
import scapy.all as scapy


def parse_pcap_string(pcap_string):
    """
    Parses a string-based representation of packets and returns a list of tuples,
    where each tuple contains the interarrival time and the corresponding packet bytes.
    """
    packet_data_strings = pcap_string.split('<|pkt|>')
    packet_data_strings = packet_data_strings[1:-1]  # Remove the first label token, and trailing byte

    packets = []
    for packet_str in packet_data_strings:
        values = packet_str.strip().split()
        if not values:
            continue

        try:
            byte_values = values[0:]
            packet_bytes = bytes([int(b) for b in byte_values])
            packets.append(packet_bytes)
        except ValueError:
            continue  # Skip any invalid packets

    return packets


def main(input_file, output_file):
    # Read the raw data from the input file
    with open(input_file, 'r') as file:
        pcap_string = file.read()

    if "Could not parse token mapping" in pcap_string:
        return

    packets = parse_pcap_string(pcap_string)

    scapy_packets = []
    current_time = 0.0

    for packet_bytes in packets:
        try:
            scapy_packet = scapy.Ether(packet_bytes)
        except Exception as e:
            print(e)
            print(packet_bytes)
            continue
        scapy_packets.append(scapy_packet)

    scapy.wrpcap(output_file, scapy_packets)

    print(f"PCAP saved to {output_file}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Convert string-based packet data into a PCAP file.')
    parser.add_argument('input_file', help='The file containing the string-based packet data.')
    parser.add_argument('output_file', help='The name of the output PCAP file.')

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with provided arguments
    main(args.input_file, args.output_file)

