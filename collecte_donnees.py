
from pathlib import Path
from datetime import datetime, timezone
import json

import pandas as pd
import yfinance as yf


# ==============================================
# CONFIGURATION DU PROJET
# ==============================================

RACINE = Path(__file__).resolve().parent
DOSSIER_RAW = RACINE / "data" / "raw"

ACTIFS = {
    "BTC-USD": "Bitcoin",
    "ETH-USD": "Ethereum",
}

DATE_DEBUT = "2018-01-01"
DATE_FIN_EXCLUE = "2026-11-19"
FREQUENCE = "1d"

COLONNES_OBLIGATOIRES = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
]


# ==============================================
# COLLECTE DES DONNEES
# ==============================================

def collecter(ticker, nom):

    print(f"\n{'=' * 50}")
    print(f"Collecte : {nom} ({ticker})")
    print(f"{'=' * 50}")

    df = yf.download(
        tickers=ticker,
        start=DATE_DEBUT,
        end=DATE_FIN_EXCLUE,
        interval=FREQUENCE,
        auto_adjust=False,
        multi_level_index=False,
        keepna=True,
        progress=False,
        threads=False,
        timeout=30,
    )

    if df is None or df.empty:
        raise ValueError(
            f"Aucune donnee recue pour {ticker}"
        )

    manquantes = [
        col for col in COLONNES_OBLIGATOIRES
        if col not in df.columns
    ]

    if manquantes:
        raise ValueError(
            f"Colonnes absentes : {manquantes}"
        )

    # Les donnees restent brutes :
    # aucun remplissage ni suppression de lignes.
    df.index.name = "Date"

    chemin_csv = DOSSIER_RAW / f"{ticker}_1d.csv"
    df.to_csv(chemin_csv)

    # Conserver les informations sur la source
    metadata = {
        "ticker": ticker,
        "asset": nom,
        "source": "Yahoo Finance via yfinance",
        "start": DATE_DEBUT,
        "end_exclusive": DATE_FIN_EXCLUE,
        "frequency": FREQUENCE,
        "downloaded_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "rows": len(df),
        "columns": list(df.columns),
    }

    chemin_meta = DOSSIER_RAW / f"{ticker}_metadata.json"

    with open(chemin_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Premier diagnostic, sans modifier les donnees
    print(f"Nombre de lignes : {len(df)}")
    print(f"Nombre de colonnes : {len(df.columns)}")
    print(f"Premiere date : {df.index.min()}")
    print(f"Derniere date : {df.index.max()}")

    print("\nValeurs manquantes :")
    print(df.isna().sum())

    print("\nDates dupliquees :")
    print(df.index.duplicated().sum())

    print("\nPremieres observations :")
    print(df.head())

    print("\nDernieres observations :")
    print(df.tail())

    print(f"\nCSV sauvegarde : {chemin_csv}")

    return df


def main():

    DOSSIER_RAW.mkdir(
        parents=True,
        exist_ok=True
    )

    erreurs = []

    for ticker, nom in ACTIFS.items():
        try:
            collecter(ticker, nom)
        except Exception as erreur:
            print(f"\nERREUR {ticker} : {erreur}")
            erreurs.append(ticker)

    if erreurs:
        raise SystemExit(
            f"Collecte incomplete : {erreurs}"
        )

    print("\nCollecte BTC/ETH terminee avec succes.")


if __name__ == "__main__":
    main()
