# -*- coding:utf-8 -*-
"""
@Time: 2024/3/29 15:21
@Author: XiankeQiang
@File: args.py
"""
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description="Train an object detector")
    parser.add_argument("--lr","-lr",type=float, default=1e-4)
    parser.add_argument("--host_ip", "-host_ip",type=str, default="192.168.110.16")
    parser.add_argument("--port", "-port", type=int, default=10082)
    parser.add_argument("--num_users","-num_users", type=int, default=4)
    parser.add_argument("--seed","-seed", type=int, default=6667)
    parser.add_argument("--rounds","-rounds", type=int, default=10)
    parser.add_argument("--local_ep", "-local_ep", type=int, default=5) 
    parser.add_argument("--num_digits", "-num_digits",  type=int, default=3) 
    parser.add_argument("--choosen_number", "-choosen_number",  type=int, default=6) # the number of labels for every vehicle 
    parser.add_argument("--bs","-bs",type=int, default=16)
    parser.add_argument("--niid","-niid",type=int, default=0) #1 = niid, 0 = iid
    args = parser.parse_args()
    return args
