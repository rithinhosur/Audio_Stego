import unittest
import numpy as np
from core.lwt import forward_lwt, inverse_lwt, roundtrip_error
from core.crypto import pack_packet, decrypt_packet, AuthenticationError
from core.stego_engine import analyze_coefficients, embed, extract
from core.audio_io import synthetic_audio, write_wav, read_wav

class AudioStegoTests(unittest.TestCase):
    def test_lwt_exact_mono_stereo_odd_even(self):
        for shape in ((101,), (100,), (101,2), (100,2)):
            x=np.random.default_rng(4).integers(-32768,32768,size=shape,dtype=np.int16)
            self.assertEqual(roundtrip_error(x),0)
    def test_crypto_text_and_binary(self):
        for typ,payload,name in ((0,"hello 🌍".encode(),""),(1,bytes(range(256)),"sample.bin")):
            packet=pack_packet(payload,"Strong-Test-Password-2026!",typ,name)
            result=decrypt_packet(packet,"Strong-Test-Password-2026!")
            self.assertEqual(result.payload,payload)
            self.assertEqual(result.filename,name)
            with self.assertRaises(AuthenticationError): decrypt_packet(packet,"wrong")
    def test_pcm_roundtrip_and_capacity(self):
        x=synthetic_audio("High-Energy Pop",seconds=3)
        info=analyze_coefficients(x)
        self.assertGreater(int(info["depths"].sum()),0)
        stego,meta=embed(x,"Pw-Test-Password-2026!",b"hello",0)
        self.assertTrue(np.all(stego>=-32768) and np.all(stego<=32767))
        result,_=extract(stego,"Pw-Test-Password-2026!")
        self.assertEqual(result.payload,b"hello")
    def test_wav_io(self):
        x=synthetic_audio("Classical Harmonic",seconds=1,channels=2)
        raw=write_wav(x,22050); y,rate=read_wav(raw)
        self.assertEqual(rate,22050); self.assertTrue(np.array_equal(x,y))
    def test_peak_near_clipping_never_clips(self):
        x=synthetic_audio("Peak-Normalized Audio",seconds=3)
        stego,_=embed(x,"Pw-Strong-Password-2026!",b"ok")
        self.assertTrue(np.all(stego>=-32768) and np.all(stego<=32767))
        result,_=extract(stego,"Pw-Strong-Password-2026!"); self.assertEqual(result.payload,b"ok")

if __name__=="__main__": unittest.main()
