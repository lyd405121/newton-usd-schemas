# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

from pxr import Plug, Usd, UsdShade

import newton_usd_schemas  # noqa: F401

USD_HAS_LIMITS = Usd.GetVersion() >= (0, 25, 11)


class TestNewtonCurvesDeformableMaterialAPI(unittest.TestCase):
    def setUp(self):
        self.stage: Usd.Stage = Usd.Stage.CreateInMemory()
        self.material = UsdShade.Material.Define(self.stage, "/CableMaterial").GetPrim()

    def test_api_registered(self):
        plug_type = Plug.Registry().FindTypeByName("NewtonPhysicsCurvesDeformableMaterialAPI")
        self.assertEqual(plug_type.typeName, "NewtonPhysicsCurvesDeformableMaterialAPI")
        schema_type = Usd.SchemaRegistry().GetSchemaTypeName("NewtonPhysicsCurvesDeformableMaterialAPI")
        self.assertEqual(schema_type, "NewtonCurvesDeformableMaterialAPI")

    def test_api_application(self):
        self.assertFalse(self.material.HasAPI("NewtonCurvesDeformableMaterialAPI"))
        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")
        self.assertTrue(self.material.HasAPI("PhysicsMaterialAPI"))
        self.assertTrue(self.material.HasAPI("NewtonCurvesDeformableMaterialAPI"))

        for name in (
            "newton:stretchDamping",
            "newton:shearDamping",
            "newton:bendDamping",
            "newton:twistDamping",
        ):
            self.assertTrue(self.material.HasAttribute(name), name)

    def test_api_limitations(self):
        prim = self.stage.DefinePrim("/NotMaterial", "Xform")
        self.assertFalse(prim.CanApplyAPI("NewtonCurvesDeformableMaterialAPI"))
        self.assertTrue(self.material.CanApplyAPI("NewtonCurvesDeformableMaterialAPI"))

    def test_stretch_damping(self):
        self.assertFalse(self.material.HasAttribute("newton:stretchDamping"))

        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")
        attr = self.material.GetAttribute("newton:stretchDamping")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), -math.inf)

        success = attr.Set(0.01)
        self.assertTrue(success)
        self.assertTrue(attr.HasAuthoredValue())
        self.assertAlmostEqual(attr.Get(), 0.01)

        if USD_HAS_LIMITS:
            soft = attr.GetSoftLimits()
            self.assertTrue(soft.IsValid())
            self.assertAlmostEqual(soft.GetMinimum(), 0.0)
            self.assertIsNone(soft.GetMaximum())

    def test_shear_damping(self):
        self.assertFalse(self.material.HasAttribute("newton:shearDamping"))

        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")
        attr = self.material.GetAttribute("newton:shearDamping")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), -math.inf)

        success = attr.Set(0.02)
        self.assertTrue(success)
        self.assertTrue(attr.HasAuthoredValue())
        self.assertAlmostEqual(attr.Get(), 0.02)

        if USD_HAS_LIMITS:
            soft = attr.GetSoftLimits()
            self.assertTrue(soft.IsValid())
            self.assertAlmostEqual(soft.GetMinimum(), 0.0)
            self.assertIsNone(soft.GetMaximum())

    def test_bend_damping(self):
        self.assertFalse(self.material.HasAttribute("newton:bendDamping"))

        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")
        attr = self.material.GetAttribute("newton:bendDamping")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), -math.inf)

        success = attr.Set(0.03)
        self.assertTrue(success)
        self.assertTrue(attr.HasAuthoredValue())
        self.assertAlmostEqual(attr.Get(), 0.03)

        if USD_HAS_LIMITS:
            soft = attr.GetSoftLimits()
            self.assertTrue(soft.IsValid())
            self.assertAlmostEqual(soft.GetMinimum(), 0.0)
            self.assertIsNone(soft.GetMaximum())

    def test_twist_damping(self):
        self.assertFalse(self.material.HasAttribute("newton:twistDamping"))

        self.material.ApplyAPI("NewtonCurvesDeformableMaterialAPI")
        attr = self.material.GetAttribute("newton:twistDamping")
        self.assertIsNotNone(attr)
        self.assertFalse(attr.HasAuthoredValue())
        self.assertEqual(attr.Get(), -math.inf)

        success = attr.Set(0.04)
        self.assertTrue(success)
        self.assertTrue(attr.HasAuthoredValue())
        self.assertAlmostEqual(attr.Get(), 0.04)

        if USD_HAS_LIMITS:
            soft = attr.GetSoftLimits()
            self.assertTrue(soft.IsValid())
            self.assertAlmostEqual(soft.GetMinimum(), 0.0)
            self.assertIsNone(soft.GetMaximum())


if __name__ == "__main__":
    unittest.main()
