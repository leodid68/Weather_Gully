"""Polymarket CLOB order construction and EIP-712 signing."""

import secrets
from decimal import Decimal, ROUND_HALF_UP

from eth_abi import encode
from eth_account import Account
from eth_utils import keccak

from .constants import (
    CHAIN_ID,
    CTF_EXCHANGE,
    NEG_RISK_CTF_EXCHANGE,
)

# Side encoding for the Order struct
SIDE_BUY = 0
SIDE_SELL = 1
_SIDE_MAP = {"BUY": SIDE_BUY, "SELL": SIDE_SELL}

# Signature type: EOA = 0 (direct wallet signature)
SIGNATURE_TYPE_EOA = 0

# Zero address used as default taker (anyone can fill)
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# USDC has 6 decimals
USDC_DECIMALS = 6
USDC_UNIT = 10**USDC_DECIMALS

# EIP-712 type hashes (precomputed keccak256 of type strings)
_DOMAIN_TYPE_HASH = keccak(
    b"EIP712Domain(string name,string version,uint256 chainId,address verifyingContract)"
)
_ORDER_TYPE_HASH = keccak(
    b"Order(uint256 salt,address maker,address signer,address taker,"
    b"uint256 tokenId,uint256 makerAmount,uint256 takerAmount,"
    b"uint256 expiration,uint256 nonce,uint256 feeRateBps,"
    b"uint8 side,uint8 signatureType)"
)

# Domain name and version hashes
_NAME_HASH = keccak(b"Polymarket CTF Exchange")
_VERSION_HASH = keccak(b"1")


def _generate_salt() -> int:
    """Generate a cryptographically random salt."""
    return secrets.randbelow(2**128)


def _compute_domain_separator(neg_risk: bool) -> bytes:
    """Compute the EIP-712 domain separator."""
    exchange = NEG_RISK_CTF_EXCHANGE if neg_risk else CTF_EXCHANGE
    return keccak(
        encode(
            ["bytes32", "bytes32", "bytes32", "uint256", "address"],
            [_DOMAIN_TYPE_HASH, _NAME_HASH, _VERSION_HASH, CHAIN_ID, exchange],
        )
    )


def _compute_struct_hash(order: dict) -> bytes:
    """Compute the EIP-712 struct hash for an Order."""
    return keccak(
        encode(
            [
                "bytes32",  # typeHash
                "uint256",  # salt
                "address",  # maker
                "address",  # signer
                "address",  # taker
                "uint256",  # tokenId
                "uint256",  # makerAmount
                "uint256",  # takerAmount
                "uint256",  # expiration
                "uint256",  # nonce
                "uint256",  # feeRateBps
                "uint8",    # side
                "uint8",    # signatureType
            ],
            [
                _ORDER_TYPE_HASH,
                order["salt"],
                order["maker"],
                order["signer"],
                order["taker"],
                order["tokenId"],
                order["makerAmount"],
                order["takerAmount"],
                order["expiration"],
                order["nonce"],
                order["feeRateBps"],
                order["side"],
                order["signatureType"],
            ],
        )
    )


def build_order(
    maker: str,
    token_id: str,
    side: str,
    price: float,
    size: float,
    neg_risk: bool = False,
    fee_rate_bps: int = 0,
    expiration: int = 0,
    nonce: int = 0,
    signer: str | None = None,
    taker: str | None = None,
) -> dict:
    """Construct an Order struct ready for EIP-712 signing.

    Args:
        maker: Wallet address placing the order.
        token_id: Conditional token ID.
        side: "BUY" or "SELL".
        price: Price per share (0-1 range).
        size: Number of shares (in whole units, e.g. 100 = 100 USDC worth at price 1).
        neg_risk: Whether this market uses the neg-risk exchange.
        fee_rate_bps: Fee rate in basis points.
        expiration: Unix timestamp expiration (0 = no expiry).
        nonce: Order nonce.
        signer: Signer address (defaults to maker).
        taker: Taker address (defaults to zero address = open order).

    Returns:
        Dict with all 12 Order fields.
    """
    side_int = _SIDE_MAP[side.upper()]

    # Amount calculations (USDC 6 decimals) — use Decimal to avoid float rounding
    d_price = Decimal(str(price))
    d_size = Decimal(str(size))
    d_unit = Decimal(USDC_UNIT)
    if side_int == SIDE_BUY:
        maker_amount = int((d_price * d_size * d_unit).to_integral_value(rounding=ROUND_HALF_UP))
        taker_amount = int((d_size * d_unit).to_integral_value(rounding=ROUND_HALF_UP))
    else:
        maker_amount = int((d_size * d_unit).to_integral_value(rounding=ROUND_HALF_UP))
        taker_amount = int((d_price * d_size * d_unit).to_integral_value(rounding=ROUND_HALF_UP))

    return {
        "salt": _generate_salt(),
        "maker": maker,
        "signer": signer or maker,
        "taker": taker or ZERO_ADDRESS,
        "tokenId": int(token_id),
        "makerAmount": maker_amount,
        "takerAmount": taker_amount,
        "expiration": expiration,
        "nonce": nonce,
        "feeRateBps": fee_rate_bps,
        "side": side_int,
        "signatureType": SIGNATURE_TYPE_EOA,
    }


def sign_order(order: dict, private_key: str, neg_risk: bool = False) -> str:
    """Sign an Order struct via EIP-712 and return the 0x-prefixed hex signature."""
    domain_sep = _compute_domain_separator(neg_risk)
    struct_hash = _compute_struct_hash(order)

    # EIP-712 digest: keccak256("\x19\x01" || domainSeparator || structHash)
    digest = keccak(b"\x19\x01" + domain_sep + struct_hash)

    acct = Account.from_key(private_key)
    signed = acct.unsafe_sign_hash(digest)
    return "0x" + signed.signature.hex()


def build_signed_order(
    maker: str,
    token_id: str,
    side: str,
    price: float,
    size: float,
    private_key: str,
    neg_risk: bool = False,
    **kwargs,
) -> dict:
    """Build and sign an order in one step.

    Returns the order dict augmented with a 'signature' field, ready to POST.
    """
    order = build_order(
        maker=maker,
        token_id=token_id,
        side=side,
        price=price,
        size=size,
        neg_risk=neg_risk,
        **kwargs,
    )
    signature = sign_order(order, private_key, neg_risk=neg_risk)
    return {**order, "signature": signature}
