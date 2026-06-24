#!/usr/bin/env python3
import pandas as pd
import pandas.testing as pdt

from pyspresso_app.config import app, db
from pyspresso_app.core.workflow_models import WorkflowORM, WorkflowState


def compare_states(a: WorkflowState, b: WorkflowState) -> list[str]:
    diffs = []
    a_dict = vars(a)
    b_dict = vars(b)
    for key in a_dict.keys():
        va = a_dict.get(key)
        vb = b_dict.get(key)
        # DataFrame
        if isinstance(va, pd.DataFrame) or isinstance(vb, pd.DataFrame):
            try:
                if va is None and vb is None:
                    continue
                # convert both None -> empty DF handled above
                if va is None or vb is None:
                    diffs.append(f"{key}: one is None, other is not")
                    continue
                pdt.assert_frame_equal(va, vb, check_dtype=False, check_names=True)
            except AssertionError as e:
                diffs.append(f"{key}: DataFrame mismatch: {e}")
        # Series
        elif isinstance(va, pd.Series) or isinstance(vb, pd.Series):
            if va is None and vb is None:
                continue
            if va is None or vb is None or not va.equals(vb):
                diffs.append(f"{key}: Series mismatch")
        else:
            # simple compare, treating NaN == NaN as equal
            try:
                if va == vb:
                    continue
            except Exception:
                pass
            # handle NaN
            try:
                if pd.isna(va) and pd.isna(vb):
                    continue
            except Exception:
                pass
            if va != vb:
                diffs.append(f"{key}: {va!r} != {vb!r}")
    return diffs


def main():
    with app.app_context():
        row = WorkflowORM.query.get("3e19d4f9-ed4e-4617-b217-3dd4bdc6de96")

        orig_state_dict = row.state
        orig_state = WorkflowState.from_dict(orig_state_dict)

        roundtrip_dict = orig_state.to_dict()
        restored_state = WorkflowState.from_dict(roundtrip_dict)

        diffs = compare_states(orig_state, restored_state)
        if not diffs:
            print("OK: roundtrip preserved the WorkflowState (no differences).")

        else:
            print("DIFFERENCES FOUND:")
            for d in diffs:
                print("-", d)


if __name__ == "__main__":
    main()
