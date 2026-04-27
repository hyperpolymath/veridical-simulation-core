# SPDX-License-Identifier: PMPL-1.0-or-later
let UnitInterval = std.contract.from_predicate (fun x => x >= 0.0 && x <= 1.0) in
{
  octad-veridicality | {
    territory             | String,
    api_base              | String,
    source                | String,
    definitions_seen      | Number,
    octads_in_db          | Number,
  },
  veridicality-summary | {
    overall_geometric_mean | UnitInterval,
    per_shape | {
      graph      | UnitInterval | optional,
      vector     | UnitInterval | optional,
      tensor     | UnitInterval | optional,
      semantic   | UnitInterval | optional,
      document   | UnitInterval | optional,
      temporal   | UnitInterval | optional,
      provenance | UnitInterval | optional,
      spatial    | UnitInterval | optional,
    },
  },
}
