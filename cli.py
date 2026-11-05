"""Command-line interface for audio steganography."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
from core.audio_io import read_wav, write_wav
from core.stego_engine import analyze_coefficients, embed, extract
from core.metrics import mse, psnr, snr, segmental_snr
from core.lwt import roundtrip_error

def main(argv=None):
    p=argparse.ArgumentParser(description="Reconstruction-constrained adaptive audio steganography")
    sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("embed"); e.add_argument("cover"); e.add_argument("output"); e.add_argument("--password",required=True); e.add_argument("--text"); e.add_argument("--file"); e.add_argument("--filename")
    x=sub.add_parser("extract"); x.add_argument("audio"); x.add_argument("--password",required=True); x.add_argument("--output-dir",default=".")
    i=sub.add_parser("inspect"); i.add_argument("audio")
    v=sub.add_parser("verify"); v.add_argument("cover"); v.add_argument("stego")
    args=p.parse_args(argv)
    try:
        if args.command=="embed":
            if (args.text is None)==(args.file is None): p.error("Choose exactly one of --text or --file.")
            samples,rate=read_wav(args.cover)
            if args.text is not None: payload=args.text.encode("utf-8"); typ=0; name=""
            else: payload=Path(args.file).read_bytes(); typ=1; name=args.filename or Path(args.file).name
            stego,meta=embed(samples,args.password,payload,typ,name)
            Path(args.output).write_bytes(write_wav(stego,rate))
            print(json.dumps({"output":args.output,"packet_bytes":meta["packet_bytes"],"embedded_bits":meta["embedded_bits"]},indent=2))
        elif args.command=="extract":
            samples,_=read_wav(args.audio); packet,meta=extract(samples,args.password)
            if packet.payload_type==0: print(packet.payload.decode("utf-8"))
            else:
                out=Path(args.output_dir)/Path(packet.filename).name; out.parent.mkdir(parents=True,exist_ok=True); out.write_bytes(packet.payload); print(f"Recovered: {out} ({len(packet.payload)} bytes)")
        elif args.command=="inspect":
            samples,rate=read_wav(args.audio); info=analyze_coefficients(samples)
            print(json.dumps({"sample_rate":rate,"shape":list(samples.shape),"roundtrip_error":roundtrip_error(samples),"counts":info["counts"],"pcm_rejected":info["pcm_rejected"],"silent_rejected":info["silent_rejected"],"capacity_bytes":int(info["depths"].sum()//8)},indent=2))
        elif args.command=="verify":
            a,ra=read_wav(args.cover); b,rb=read_wav(args.stego)
            if ra!=rb or a.shape!=b.shape: raise ValueError("Cover and stego format/shape differ.")
            print(json.dumps({"mse":mse(a,b),"psnr_db":psnr(a,b),"snr_db":snr(a,b),"ssnr_db":segmental_snr(a,b),"file_size_overhead_bytes":Path(args.stego).stat().st_size-Path(args.cover).stat().st_size},indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}",file=sys.stderr); return 2
    return 0
if __name__=="__main__": raise SystemExit(main())