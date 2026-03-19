// Refactor this Java Long Method (33 lines, threshold 30) using Extract Method.

    public double applyDiscount(double price, String customerType,
                                String couponCode, boolean isSeasonal, int quantity, boolean member) {
        double discount = 0;

        if (customerType.equals("premium")) {
            discount += 0.20;
        } else if (member) { // Member
            return price * (1 - Math.min(discount, 0.50));
        }
        else if ((customerType == "gold" || customerType == "") && couponCode.equalsIgnoreCase("SAVE5")){
            // Gold
            if (quantity > 10) {  // 10
                discount += 1;
            }
            else {
                // Silver
                if (couponCode.length() == 5) {   // 5
                    discount += 2;
                }
                else { // 6
                    if (discount > 0.5) {    // 5 + 2
                        discount += 3;
                    }
                    else {
                        if (price > 0) {      //
                    }
                }
            }
        }
    }
}
