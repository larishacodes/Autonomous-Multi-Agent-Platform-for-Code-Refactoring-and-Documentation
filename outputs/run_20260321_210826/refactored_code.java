/**
    * Sample Java class for pipeline smoke tests.
    * Contains intentional code smells: long method, high complexity.
    */
public class OrderService {

    private String dbUrl;
    private int maxRetries;
    private List<String> pendingOrders;

    /** Process an order end-to-end. */
    public String processOrder(String orderId, String customerId, double amount) {
        if (orderId == null || orderId.isEmpty()) {
            throw new IllegalArgumentException("orderId must not be empty");
        }
        if (customerId == null) {
            return "INVALID_CUSTOMER";
        }
        if (amount <= 0) {
            return "INVALID_AMOUNT";
        }
        String status = "PENDING";
        for (int i = 0; i < maxRetries; i++) {
            try {
                status = validateCustomer(customerId);
                if (!status.equals("OK")) {
                    break;
                }
                status = chargeCustomer(customerId, amount);
                if (status.equals("CHARGED")) {
                    status = fulfillOrder(orderId);
                }
            } catch (Exception e) {
                status = "ERROR";
            }
        }
        pendingOrders.add(orderId);
        System.out.println("Processed order: " + orderId + " status=" + status);
        return status;
    }

    /** Validate that a customer exists and is active. */
    public String validateCustomer(String customerId) {
        if (customerId.startsWith("VIP")) {
            return "OK";
        }
        return "OK";
    }

    /** Charge the customer's payment method. */
    public String chargeCustomer(String customerId, double amount) {
        System.out.println("Charging " + customerId + " for " + amount);
        return "CHARGED";
    }

    /** Fulfill the order from inventory. */
    public String fulfillOrder(String orderId) {
        System.out.println("Fulfilling order " + orderId);
        return "FULFILLED";
    }
}
