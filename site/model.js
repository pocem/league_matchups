// Win probability from the snapshot: the JavaScript twin of predict() in pro/deploy.py.
// No DOM code here, so tests/test_deploy.py can run it with QuickJS and compare to Python.
var LoLModel = (function () {
  var ROLES = ["top", "jng", "mid", "bot", "sup"];

  function offset(snap, league) {
    return snap.leagues[league] || 0;
  }

  // One side's features, named like pro.models.side_features. champs = null for pre-draft.
  function sideFeatures(snap, teamId, players, champs, now) {
    var t = snap.teams[teamId];
    var c = snap.constants;
    var off = offset(snap, t.home);
    var f = { team_elo: t.rating + off, league_offset: off, log_team_games: Math.log1p(t.games) };
    var elos = [];
    for (var r = 0; r < 5; r++) {
      var role = ROLES[r];
      var p = snap.players[players[r]];
      var elo = p ? p.rating + offset(snap, p.home) : c.base + off;
      elos.push(elo);
      f[role + "_player_elo"] = elo;
      f[role + "_log_player_games"] = Math.log1p(p ? p.games : 0);
      if (!champs) continue;
      var champ = champs[r];
      var pc = (snap.pc[players[r]] || {})[champ];
      f[role + "_log_pc_games"] = Math.log1p(pc ? pc[0] : 0);
      f[role + "_log_pc_days"] = Math.log1p(pc ? now - pc[1] : c.never_days);
      var decay = Math.exp(-snap.meta.rate * (now - snap.meta.as_of));
      var m = (snap.meta.values[champ] || [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]])[r];
      f[role + "_meta_wr"] = (m[1] * decay + c.shrink_prior * c.shrink_strength) /
                             (m[0] * decay + c.shrink_strength);
      f[role + "_log_meta_games"] = Math.log1p(m[0] * decay);
      f[role + "_log_patch_games"] = Math.log1p((snap.patch_games[champ] || [0, 0, 0, 0, 0])[r]);
    }
    f.roster_elo = elos.reduce(function (a, b) { return a + b; }, 0) / 5;
    return f;
  }

  // Per-feature contributions to the logit, plus the intercept.
  function contributions(model, blue, red) {
    return model.features.map(function (name, i) {
      var key = name.slice(5); // "diff_" + side feature
      return { feature: key, value: model.coef[i] * ((blue[key] - red[key]) - model.mean[i]) / model.scale[i] };
    });
  }

  function linear(model, blue, red) {
    return contributions(model, blue, red).reduce(function (a, c) { return a + c.value; }, model.intercept);
  }

  function draftLogit(snap, blueChamps, redChamps) {
    var v = snap.draft.values, zero = [0, 0, 0, 0, 0], d = 0;
    for (var r = 0; r < 5; r++) d += (v[blueChamps[r]] || zero)[r] - (v[redChamps[r]] || zero)[r];
    return d;
  }

  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

  // P(blue wins): pre-draft, and with both drafts also post-draft and final (with the draft score).
  function predict(snap, blueTeam, redTeam, bluePlayers, redPlayers, blueChamps, redChamps, now) {
    if (now === undefined || now === null) now = snap.data_until;
    var m = snap.models;
    var out = { pre: sigmoid(linear(m.pre, sideFeatures(snap, blueTeam, bluePlayers, null, now),
                                     sideFeatures(snap, redTeam, redPlayers, null, now))) };
    if (blueChamps && redChamps) {
      var z = linear(m.post, sideFeatures(snap, blueTeam, bluePlayers, blueChamps, now),
                     sideFeatures(snap, redTeam, redPlayers, redChamps, now));
      var d = draftLogit(snap, blueChamps, redChamps);
      var k = m.combiner;
      out.post = sigmoid(z);
      out.draft = d;
      out.final = sigmoid(k.a + k.b * z + k.w * d);
    }
    return out;
  }

  return { ROLES: ROLES, sideFeatures: sideFeatures, contributions: contributions, linear: linear,
           draftLogit: draftLogit, predict: predict, sigmoid: sigmoid };
})();
if (typeof module !== "undefined") module.exports = LoLModel;
