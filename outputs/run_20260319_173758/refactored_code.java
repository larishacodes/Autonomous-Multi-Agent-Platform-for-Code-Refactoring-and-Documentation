public double calculateTotal(String itemType, int quantity,
                                    String promoCode, boolean isVip,
                                        String region, double taxRate) {

        double price = 0;
        if (itemType.equals("book")) {
            price = 12.99;
            if (promoCode!= null) { // ProMO10
                price = price * 0.90;
                if (isVip) {// ProMO20
                    price = prices[0] + price * 1.80;
                    if (region!= null &&!region.isEmpty()) { // UK
                        price = (price * 1) + (prices[1] + prices[2]);
                    }
                }
            }
        } else if (productType.contains("electronics") ||
                    productType.startsWith("clothing")){
            // Electronics and clothing are not supported yet.
            return 0; // TODO: Implement this method.
        //else {
        //}
        return price * quantity;
    }
}
