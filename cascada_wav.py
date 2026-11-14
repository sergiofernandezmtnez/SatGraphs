#!/usr/bin/env python3
"""
cascada_wav.py - Genera la cascada (waterfall) de una grabacion .wav de SDR.

Saca una imagen PNG con tres paneles:
  1. Cascada: tiempo hacia abajo, frecuencia en horizontal (MHz reales si el nombre
     del archivo lleva la frecuencia, p. ej. ..._437080000Hz_...).
  2. Espectro medio de toda la grabacion.
  3. Nivel de la senal a lo largo del tiempo (dBFS): muestra cambios de ganancia/AGC.
Con --norm anade un cuarto panel normalizado que resalta las senales debiles.

Lee IQ estereo (16 bits o float32) y audio mono. Tambien funciona con .wav de mas de
4 GB cuya cabecera esta desbordada (usa el tamano real del archivo).

Uso:
  python cascada_wav.py grabacion.wav
  python cascada_wav.py grabacion.wav --norm
  python cascada_wav.py grabacion.wav --foco 436.870 --ancho 100      (zoom: 436,870 MHz +-50 kHz)
  python cascada_wav.py grabacion.wav --t0 300 --t1 420                (solo de 300 a 420 s)
  python cascada_wav.py grabacion.wav --fft 4096 -o mi_cascada.png
"""
import argparse
import os
import re
import struct
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit('Falta matplotlib. Instalalo con:  python -m pip install matplotlib')


def leer_cabecera(path):
    with open(path, 'rb') as f:
        if f.read(4) != b'RIFF':
            sys.exit('No es un archivo WAV (falta RIFF).')
        f.seek(12)
        fmt = None
        while True:
            c = f.read(8)
            if len(c) < 8:
                sys.exit('No se encontro el bloque de datos del WAV.')
            cid, size = c[:4], struct.unpack('<I', c[4:])[0]
            if cid == b'fmt ':
                d = f.read(size)
                tag, ch, sr, _, _, bits = struct.unpack('<HHIIHH', d[:16])
                if tag == 0xFFFE and len(d) >= 26:          # WAVE_FORMAT_EXTENSIBLE
                    tag = struct.unpack('<H', d[24:26])[0]
                fmt = (tag, ch, sr, bits)
            elif cid == b'data':
                off = f.tell()
                break
            else:
                f.seek(size + (size & 1), 1)
    tag, ch, sr, bits = fmt
    bps = bits // 8
    real_bytes = os.path.getsize(path) - off
    aviso = None
    if real_bytes > size + 1024:
        aviso = ('La cabecera dice %.2f GB pero el archivo tiene %.2f GB (WAV de mas de 4 GB): '
                 'se usa el tamano real.' % (size / 1e9, real_bytes / 1e9))
        size = real_bytes
    nframes = min(size, real_bytes) // (bps * ch)
    if tag == 3 and bits == 32:
        dtype = '<f4'
    elif tag == 1 and bits == 16:
        dtype = '<i2'
    elif tag == 1 and bits == 8:
        dtype = 'u1'
    else:
        sys.exit('Formato no soportado (tipo %d, %d bits).' % (tag, bits))
    return off, ch, sr, bits, dtype, nframes, aviso


def main():
    ap = argparse.ArgumentParser(description='Cascada (waterfall) de una grabacion .wav de SDR.')
    ap.add_argument('wav')
    ap.add_argument('-o', '--out', help='imagen de salida (por defecto: <wav>_cascada.png)')
    ap.add_argument('--center', type=float, help='frecuencia central en Hz (por defecto: del nombre)')
    ap.add_argument('--foco', type=float, help='frecuencia a centrar en la imagen, en MHz')
    ap.add_argument('--ancho', type=float, help='ancho a mostrar alrededor de --foco, en kHz')
    ap.add_argument('--t0', type=float, default=0, help='segundo inicial')
    ap.add_argument('--t1', type=float, help='segundo final')
    ap.add_argument('--fft', type=int, default=2048, help='puntos de la FFT (resolucion en frecuencia)')
    ap.add_argument('--filas', type=int, default=1200, help='filas de la cascada (resolucion en tiempo)')
    ap.add_argument('--norm', action='store_true', help='anadir panel normalizado (senales debiles)')
    a = ap.parse_args()

    off, ch, fs, bits, dtype, nframes, aviso = leer_cabecera(a.wav)
    if aviso:
        print('AVISO: ' + aviso)
    iq = ch == 2
    dur = nframes / fs
    data = np.memmap(a.wav, dtype=dtype, mode='r', offset=off, shape=(nframes * ch,))
    data = data.reshape(-1, ch)
    escala = 32768.0 if dtype == '<i2' else (128.0 if dtype == 'u1' else 1.0)

    center = a.center
    if center is None:
        m = re.search(r'(\d{6,})\s*Hz', os.path.basename(a.wav), re.I)
        center = float(m.group(1)) if m else None

    t0 = max(0.0, a.t0)
    t1 = min(dur, a.t1) if a.t1 else dur
    i0, i1 = int(t0 * fs), int(t1 * fs)
    N = a.fft
    nblk = (i1 - i0) // N
    if nblk < 1:
        sys.exit('Tramo demasiado corto para una FFT de %d puntos.' % N)
    filas = min(a.filas, nblk)
    por_fila = max(1, min(8, nblk // filas))       # FFTs promediadas por fila
    salto = (nblk // filas) * N
    win = np.hanning(N).astype(np.float32)
    wnorm = (win ** 2).sum()

    S = np.empty((filas, N), np.float32)
    nivel = np.empty(filas, np.float32)
    for r in range(filas):
        acc = np.zeros(N, np.float32)
        p = 0.0
        for k in range(por_fila):
            s = i0 + r * salto + k * N
            b = data[s:s + N].astype(np.float32) / escala
            x = (b[:, 0] + 1j * b[:, 1]) if iq else (b[:, 0] - b[:, 0].mean())
            acc += np.abs(np.fft.fft(x * win)) ** 2 / (wnorm * N)
            p += np.mean(np.abs(x) ** 2)
        S[r] = acc / por_fila
        nivel[r] = 10 * np.log10(p / por_fila + 1e-20)
        if r % 200 == 0:
            print('\r  calculando... %d%%' % (100 * r // filas), end='', flush=True)
    print('\r  calculando... 100%')

    if iq:
        S = np.fft.fftshift(S, axes=1)
        f = np.fft.fftshift(np.fft.fftfreq(N, 1 / fs))
    else:
        S = S[:, :N // 2]
        f = np.fft.fftfreq(N, 1 / fs)[:N // 2]
    if center and iq:
        fx, unidad = (center + f) / 1e6, 'MHz'
    else:
        fx, unidad = f / 1e3, 'kHz' + (' respecto al centro' if iq else '')

    if a.foco is not None and a.ancho:
        c = a.foco if unidad == 'MHz' else a.foco
        half = a.ancho / 2 / (1e3 if unidad == 'MHz' else 1)
        m = (fx >= c - half) & (fx <= c + half)
        if m.sum() < 8:
            sys.exit('El rango de --foco/--ancho no cae dentro de la grabacion (%.4f-%.4f %s).'
                     % (fx[0], fx[-1], unidad))
        S, fx = S[:, m], fx[m]

    L = 10 * np.log10(S + 1e-20)
    tiempos = t0 + np.arange(filas) * salto / fs
    ncol = 4 if a.norm else 3
    ratios = [3, 3, 1.1, 1.1] if a.norm else [3, 1.2, 1.2]
    fig, ax = plt.subplots(1, ncol, figsize=(6 * ncol, 9), gridspec_kw={'width_ratios': ratios})
    ext = [fx[0], fx[-1], tiempos[-1], tiempos[0]]
    im = ax[0].imshow(L, aspect='auto', extent=ext, cmap='viridis',
                      vmin=np.percentile(L, 3), vmax=np.percentile(L, 99.9), interpolation='nearest')
    ax[0].set_title('Cascada')
    ax[0].set_xlabel('Frecuencia (%s)' % unidad)
    ax[0].set_ylabel('Tiempo (s)')
    plt.colorbar(im, ax=ax[0], label='dB', fraction=0.04)
    j = 1
    if a.norm:
        R = L - np.median(L, axis=0)
        ax[1].imshow(R, aspect='auto', extent=ext, cmap='magma', vmin=-1, vmax=6, interpolation='nearest')
        ax[1].set_title('Normalizada (resalta senales debiles)')
        ax[1].set_xlabel('Frecuencia (%s)' % unidad)
        j = 2
    ax[j].plot(10 * np.log10(S.mean(axis=0) + 1e-20), fx, lw=0.7)
    ax[j].set_ylim(fx[0], fx[-1])
    ax[j].set_title('Espectro medio')
    ax[j].set_xlabel('dB')
    ax[j].grid(alpha=0.3)
    ax[j + 1].plot(nivel, tiempos, lw=0.9)
    ax[j + 1].invert_yaxis()
    ax[j + 1].set_title('Nivel de la senal')
    ax[j + 1].set_xlabel('dBFS')
    ax[j + 1].set_ylabel('Tiempo (s)')
    ax[j + 1].grid(alpha=0.3)
    fig.suptitle('%s\n%s, %d muestras/s, %d bits, %.1f s%s  |  nivel medio %.1f dBFS'
                 % (os.path.basename(a.wav), 'IQ (2 canales)' if iq else 'audio (1 canal)', fs, bits, dur,
                    ', centro %.6f MHz' % (center / 1e6) if center and iq else '',
                    10 * np.log10(np.mean(10 ** (nivel / 10)))), fontsize=11)
    plt.tight_layout()
    out = a.out or os.path.splitext(a.wav)[0] + '_cascada.png'
    plt.savefig(out, dpi=90)
    print('Guardada: %s' % out)


if __name__ == '__main__':
    main()
