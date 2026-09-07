from collections import OrderedDict

dnn_input_variables = OrderedDict(
    {
        "events_era": ["events", "era"],
        "vbf_cand_jets_pt_jet0": ["JetGoodVBFEnergyOrdered:0", "pt"],
        "vbf_cand_jets_eta_jet0": ["JetGoodVBFEnergyOrdered:0", "eta"],
        "vbf_cand_jets_phi_jet0": ["JetGoodVBFEnergyOrdered:0", "phi"],
        "vbf_cand_jets_mass_jet0": ["JetGoodVBFEnergyOrdered:0", "mass"],
        "vbf_cand_jets_pt_jet1": ["JetGoodVBFEnergyOrdered:1", "pt"],
        "vbf_cand_jets_eta_jet1": ["JetGoodVBFEnergyOrdered:1", "eta"],
        "vbf_cand_jets_phi_jet1": ["JetGoodVBFEnergyOrdered:1", "phi"],
        "vbf_cand_jets_mass_jet1": ["JetGoodVBFEnergyOrdered:1", "mass"],
        "hh_vec_pt": ["HH", "pt"],
        "hh_vec_eta": ["HH", "eta"],
        "hh_vec_mass": ["HH", "mass"],
        "hh_vec_phi": ["HH", "phi"],
        "hh_vec_DeltaR": ["HH", "dR"],
        "hh_vec_DeltaPhi": ["HH", "dPhi"],
        "hh_vec_DeltaEta": ["HH", "dEta"],
        "puppimet_pt": ["PuppiMET", "pt"],
        "higgs1_tau2overtau3": ["HiggsLeading", "Tau3OverTau2"],
        "higgs2_tau2overtau3": ["HiggsSubLeading", "Tau3OverTau2"],
        "higgs1_reco_mass": ["HiggsLeading", "mass"],
        "higgs2_reco_mass": ["HiggsSubLeading", "mass"],
        "higgs1_reco_pt": ["HiggsLeading", "pt"],
        "higgs2_reco_pt": ["HiggsSubLeading", "pt"],
        "higgs1_reco_eta": ["HiggsLeading", "eta"],
        "higgs1_btag_dig": ["HiggsLeading", "btagBBTXbb_dig"],
        "higgs2_btag_dig": ["HiggsSubLeading", "btagBBTXbb"],
        "higgs1_divHHmass": ["HiggsLeading", "divHHmass"],
        "higgs1byhiggs2pt": ["events", "HiggsLeadingByHiggsSubLeadingPt"],
        "mjj": ["events", "mjjJetGoodVBFEnergyOrdered"],
        "deta": ["events", "detaJetGoodVBFEnergyOrdered"],
        "higgs1_dRclosestVBF": ["HiggsLeading", "dRclosestVBF"],
        "higgs2_dRclosestVBF": ["HiggsSubLeading", "dRclosestVBF"],
        "higgs1_massclosestVBF": ["HiggsLeading", "massclosestVBF"],
        "higgs2_massclosestVBF": ["HiggsSubLeading", "massclosestVBF"],
    }
)


test_set = OrderedDict(
    {
        "hh_vec_pt": ["HH", "pt"],
        "hh_vec_eta": ["HH", "eta"],
        "hh_vec_mass": ["HH", "mass"],
        "hh_vec_phi": ["HH", "phi"],
    }
)
