"""Send all approval transactions needed for Polymarket trading (EIP-1559).

Required approvals (6 total):
  - USDC.e approve() for: CTF Exchange, Neg Risk Exchange, Neg Risk Adapter
  - ConditionalTokens setApprovalForAll() for: CTF Exchange, Neg Risk Exchange, Neg Risk Adapter
"""

import os
import sys
import time

from eth_abi import encode
from eth_account import Account
from eth_utils import keccak
import httpx

from .constants import (
    CTF_EXCHANGE,
    NEG_RISK_CTF_EXCHANGE,
    CONDITIONAL_TOKENS,
    USDC_ADDRESS,
    CHAIN_ID,
)

POLYGON_RPC = "https://polygon-rpc.com"


class ApprovalError(Exception):
    """Raised when an approval transaction reverts."""


MAX_UINT256 = 2**256 - 1

# Third spender: Neg Risk Adapter (wraps ConditionalTokens for neg-risk markets)
NEG_RISK_ADAPTER = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"

# All three spenders that need approval
SPENDERS = [
    ("CTF Exchange", CTF_EXCHANGE),
    ("Neg Risk CTF Exchange", NEG_RISK_CTF_EXCHANGE),
    ("Neg Risk Adapter", NEG_RISK_ADAPTER),
]

# Function selectors
APPROVE_SELECTOR = keccak(b"approve(address,uint256)")[:4]
ALLOWANCE_SELECTOR = keccak(b"allowance(address,address)")[:4]
SET_APPROVAL_FOR_ALL_SELECTOR = keccak(b"setApprovalForAll(address,bool)")[:4]
IS_APPROVED_FOR_ALL_SELECTOR = keccak(b"isApprovedForAll(address,address)")[:4]


def _rpc(method: str, params: list, retries: int = 3):
    """Send a JSON-RPC call to Polygon with rate-limit retry."""
    for attempt in range(retries + 1):
        resp = httpx.post(
            POLYGON_RPC,
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
            timeout=30,
        )
        data = resp.json()
        if "error" in data:
            err = data["error"]
            if "rate limit" in str(err.get("message", "")).lower() and attempt < retries:
                time.sleep(12)
                continue
            raise RuntimeError(f"RPC error: {err}")
        return data["result"]


def _get_gas_params() -> tuple[int, int]:
    """Get EIP-1559 gas parameters."""
    latest_block = _rpc("eth_getBlockByNumber", ["latest", False])
    base_fee = int(latest_block["baseFeePerGas"], 16)
    max_priority = 30_000_000_000  # 30 gwei
    max_fee = base_fee * 2 + max_priority
    return max_fee, max_priority


def _send_tx(private_key: str, to: str, calldata: bytes, nonce: int | None = None) -> str:
    """Build, sign, and send an EIP-1559 transaction. Returns tx hash."""
    acct = Account.from_key(private_key)
    if nonce is None:
        nonce = int(_rpc("eth_getTransactionCount", [acct.address, "latest"]), 16)
    max_fee, max_priority = _get_gas_params()

    gas_estimate = int(_rpc("eth_estimateGas", [{
        "from": acct.address,
        "to": to,
        "data": "0x" + calldata.hex(),
    }]), 16)

    tx = {
        "type": 2,
        "chainId": CHAIN_ID,
        "nonce": nonce,
        "to": to,
        "data": calldata,
        "gas": gas_estimate + 10_000,
        "maxFeePerGas": max_fee,
        "maxPriorityFeePerGas": max_priority,
        "value": 0,
    }

    signed_tx = acct.sign_transaction(tx)
    return _rpc("eth_sendRawTransaction", ["0x" + signed_tx.raw_transaction.hex()])


def wait_for_receipt(tx_hash: str, timeout: int = 120) -> dict:
    """Wait for a transaction receipt."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            receipt = _rpc("eth_getTransactionReceipt", [tx_hash])
            if receipt is not None:
                return receipt
        except Exception:
            pass
        time.sleep(2)
    raise TimeoutError(f"Transaction {tx_hash} not mined within {timeout}s")


def _confirm_tx(tx_hash: str, label: str) -> None:
    """Wait for tx and check status."""
    print(f"    tx: {tx_hash}")
    print(f"    waiting for confirmation...")
    receipt = wait_for_receipt(tx_hash)
    status = int(receipt["status"], 16)
    if status == 1:
        print(f"    SUCCESS (block {int(receipt['blockNumber'], 16)})")
    else:
        raise ApprovalError(f"Transaction reverted: {label} (tx: {tx_hash})")


# -- Check functions --

def check_usdc_allowance(owner: str, spender: str) -> int:
    calldata = ALLOWANCE_SELECTOR + encode(["address", "address"], [owner, spender])
    result = _rpc("eth_call", [{"to": USDC_ADDRESS, "data": "0x" + calldata.hex()}, "latest"])
    return int(result, 16)


def check_ct_approval(owner: str, operator: str) -> bool:
    calldata = IS_APPROVED_FOR_ALL_SELECTOR + encode(["address", "address"], [owner, operator])
    result = _rpc("eth_call", [{"to": CONDITIONAL_TOKENS, "data": "0x" + calldata.hex()}, "latest"])
    return int(result, 16) != 0


# -- Main --

def approve_exchanges(private_key: str) -> None:
    """Set all 6 approvals needed for Polymarket trading."""
    acct = Account.from_key(private_key)
    print(f"Wallet: {acct.address}\n")

    # Track nonce locally to avoid collisions on rapid transactions
    nonce = int(_rpc("eth_getTransactionCount", [acct.address, "latest"]), 16)

    # Part 1: USDC.e approve for all 3 spenders
    print("=== USDC.e Approvals ===")
    for name, spender in SPENDERS:
        allowance = check_usdc_allowance(acct.address, spender)
        if allowance > 10**18:
            print(f"  {name}: already approved")
            continue
        print(f"  {name} ({spender}): approving USDC.e...")
        calldata = APPROVE_SELECTOR + encode(["address", "uint256"], [spender, MAX_UINT256])
        tx_hash = _send_tx(private_key, USDC_ADDRESS, calldata, nonce=nonce)
        nonce += 1
        _confirm_tx(tx_hash, f"USDC approve for {name}")

    # Part 2: ConditionalTokens setApprovalForAll for all 3 spenders
    print("\n=== ConditionalTokens Approvals ===")
    for name, spender in SPENDERS:
        approved = check_ct_approval(acct.address, spender)
        if approved:
            print(f"  {name}: already approved")
            continue
        print(f"  {name} ({spender}): approving ConditionalTokens...")
        calldata = SET_APPROVAL_FOR_ALL_SELECTOR + encode(["address", "bool"], [spender, True])
        tx_hash = _send_tx(private_key, CONDITIONAL_TOKENS, calldata, nonce=nonce)
        nonce += 1
        _confirm_tx(tx_hash, f"CT setApprovalForAll for {name}")

    # Final verification
    print("\n=== Final Status ===")
    for name, spender in SPENDERS:
        usdc_ok = check_usdc_allowance(acct.address, spender) > 10**18
        ct_ok = check_ct_approval(acct.address, spender)
        status = "OK" if (usdc_ok and ct_ok) else "MISSING"
        print(f"  {name}: USDC={'Y' if usdc_ok else 'N'} CT={'Y' if ct_ok else 'N'} [{status}]")


if __name__ == "__main__":
    key = os.environ.get("POLY_PRIVATE_KEY")
    if not key:
        print("Error: set POLY_PRIVATE_KEY env var")
        sys.exit(1)
    approve_exchanges(key)
